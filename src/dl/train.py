"""Train and evaluate one deep learning model for resume category classification.

Examples (run from the repo root):
  python -m src.dl.train --data data/splits --model bilstm --embed glove --glove data/glove/glove.6B.300d.txt
  python -m src.dl.train --data data/raw/Resume.csv --model textcnn --embed random
  python -m src.dl.train --data data/splits --model bilstm --rnn gru --no-attention --remove-title

Each run writes runs/<run-name>/ (best model, curves, confusion matrix, per-class report,
test predictions for error analysis) and appends one row to reports/results.csv.
"""
import argparse
import csv
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from .data import SeqDataset, Vocab, collate, load_splits
from .embeddings import glove_matrix, random_matrix
from .models import build_model
from .text import prepare

NAMES = {"bilstm": "Bi{rnn}", "textcnn": "TextCNN", "cnn_bilstm": "CNN + Bi{rnn}"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="split folder (train/val/test.csv) or raw Resume.csv")
    p.add_argument("--text-col", default="Resume_str")
    p.add_argument("--label-col", default="Category")
    p.add_argument("--model", choices=["bilstm", "textcnn", "cnn_bilstm"], required=True)
    p.add_argument("--rnn", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--no-attention", action="store_true", help="max-pooling instead of attention")
    p.add_argument("--embed", choices=["random", "glove"], default="random")
    p.add_argument("--glove", default="data/glove/glove.6B.300d.txt")
    p.add_argument("--emb-dim", type=int, default=300)
    p.add_argument("--freeze", action="store_true", help="keep the embedding layer fixed")
    p.add_argument("--remove-title", action="store_true", help="drop the leading job title (leakage test)")
    p.add_argument("--no-clean", action="store_true", help="text is already preprocessed; only tokenise")
    p.add_argument("--max-len", type=int, default=512, help="tokens kept per resume")
    p.add_argument("--truncate", choices=["head", "head_tail"], default="head")
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--filters", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5, help="early stopping on val macro-F1")
    p.add_argument("--min-epochs", type=int, default=8, help="never stop before this many epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--out", default="runs")
    p.add_argument("--results", default="reports/results.csv")
    p.add_argument("--name", default=None, help="run name (default: built from the settings)")
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(choice):
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, gold, total_loss, n = [], [], 0.0, 0
    loss_fn = nn.CrossEntropyLoss()
    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, lengths)
        total_loss += loss_fn(logits, y).item() * len(y); n += len(y)
        probs.append(torch.softmax(logits, -1).cpu()); gold.append(y.cpu())
    return torch.cat(probs).numpy(), torch.cat(gold).numpy(), total_loss / n


def run_name(a):
    parts = [a.model]
    if a.model != "textcnn":
        parts += [a.rnn, "max" if a.no_attention else "att"]
    parts += [a.embed + ("-frozen" if a.freeze else ""), "notitle" if (a.remove_title or "no_header" in a.text_col) else "title",
              f"len{a.max_len}" + ("ht" if a.truncate == "head_tail" else ""), f"s{a.seed}"]
    return "_".join(parts)


def main(argv=None):
    a = parse_args(argv)
    if "no_header" in a.text_col:  # Long's header-removed text column counts as "job title removed"
        a.remove_title = False
        a.title_removed_by_column = True
    set_seed(a.seed)
    device = pick_device(a.device)
    name = a.name or run_name(a)
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    print(f"[{name}] device={device}")

    # ---- data ----
    train_df, val_df, test_df = load_splits(a.data, a.text_col, a.label_col, a.seed)
    labels = sorted(train_df[a.label_col].unique())
    lab2id = {l: i for i, l in enumerate(labels)}
    toks = {s: [prepare(t, a.remove_title, not a.no_clean) for t in df[a.text_col]]
            for s, df in (("train", train_df), ("val", val_df), ("test", test_df))}
    vocab = Vocab.build(toks["train"], min_freq=a.min_freq)
    lens = np.array([len(t) for t in toks["train"]])
    print(f"train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}, classes={len(labels)}, "
          f"vocab={len(vocab)}, median tokens={int(np.median(lens))}, over max_len={np.mean(lens > a.max_len):.0%}")

    def loader(split, df, shuffle):
        ids = [vocab.encode(t, a.max_len, a.truncate) for t in toks[split]]
        y = [lab2id[l] for l in df[a.label_col]]
        return DataLoader(SeqDataset(ids, y), batch_size=a.batch, shuffle=shuffle, collate_fn=collate)

    train_dl, val_dl, test_dl = loader("train", train_df, True), loader("val", val_df, False), loader("test", test_df, False)

    # ---- embeddings + model ----
    coverage = None
    if a.embed == "glove":
        if not os.path.exists(a.glove):
            raise FileNotFoundError(f"GloVe file not found: {a.glove} (see README_DL.md)")
        weights, coverage = glove_matrix(a.glove, vocab, a.emb_dim, a.seed)
        print(f"GloVe coverage of vocabulary: {coverage:.1%}")
    else:
        weights = random_matrix(len(vocab), a.emb_dim, a.seed)
    cfg = dict(rnn=a.rnn, hidden=a.hidden, layers=a.layers, filters=a.filters, dropout=a.dropout,
               attention=not a.no_attention, freeze=a.freeze)
    model = build_model(a.model, weights, len(labels), cfg, pack=device.type != "mps").to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    y_train = np.array([lab2id[l] for l in train_df[a.label_col]])
    cw = compute_class_weight("balanced", classes=np.arange(len(labels)), y=y_train)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    # ---- training with early stopping ----
    history, best_f1, best_epoch, bad, t0 = [], -1.0, 0, 0, time.time()
    for epoch in range(1, a.epochs + 1):
        model.train(); tl, n = 0.0, 0
        for x, lengths, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(y); n += len(y)
        vp, vy, vl = predict(model, val_dl, device)
        vf1 = f1_score(vy, vp.argmax(1), average="macro", zero_division=0)
        vacc = accuracy_score(vy, vp.argmax(1))
        sched.step(vf1)
        history.append(dict(epoch=epoch, train_loss=tl / n, val_loss=vl, val_macro_f1=vf1, val_acc=vacc))
        print(f"epoch {epoch:2d}  train_loss {tl / n:.4f}  val_loss {vl:.4f}  val_macroF1 {vf1:.4f}  val_acc {vacc:.4f}")
        if vf1 > best_f1:
            best_f1, best_epoch, bad = vf1, epoch, 0
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience and epoch >= a.min_epochs:
                print(f"early stopping (best epoch {best_epoch})")
                break
    train_time = time.time() - t0

    # ---- test with the best checkpoint ----
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device))
    tp, ty, _ = predict(model, test_dl, device)
    pred = tp.argmax(1)
    test_f1 = f1_score(ty, pred, average="macro", zero_division=0)
    test_wf1 = f1_score(ty, pred, average="weighted", zero_division=0)
    test_acc = accuracy_score(ty, pred)
    print(f"TEST  macroF1 {test_f1:.4f}  weightedF1 {test_wf1:.4f}  acc {test_acc:.4f}  (best val macroF1 {best_f1:.4f})")

    # ---- save everything the report / demo / error analysis need ----
    pd.DataFrame(history).to_csv(os.path.join(out, "history.csv"), index=False)
    with open(os.path.join(out, "test_report.txt"), "w") as f:
        f.write(classification_report(ty, pred, target_names=labels, digits=4, zero_division=0))
    top3 = np.argsort(-tp, axis=1)[:, :3]
    pd.DataFrame({
        "row": test_df.index,
        **({"ID": test_df["ID"]} if "ID" in test_df.columns else {}),
        "true": [labels[i] for i in ty], "pred": [labels[i] for i in pred],
        "confidence": tp.max(1).round(4), "correct": ty == pred,
        "top3": ["; ".join(f"{labels[j]} {tp[r, j]:.2f}" for j in top3[r]) for r in range(len(ty))],
        "n_words": test_df[a.text_col].astype(str).str.split().str.len(),
        "text_start": test_df[a.text_col].astype(str).str[:300],
    }).to_csv(os.path.join(out, "test_predictions.csv"), index=False)
    plot_curves(history, os.path.join(out, "curves.png"), name)
    plot_confusion(confusion_matrix(ty, pred, labels=range(len(labels)), normalize="true"), labels,
                   os.path.join(out, "confusion_matrix.png"), f"{name} (test, row-normalised)")
    with open(os.path.join(out, "vocab.json"), "w") as f:
        json.dump(vocab.itos, f)
    config = dict(vars(a), labels=labels, model_cfg=cfg, vocab_size=len(vocab), params=n_params,
                  glove_coverage=coverage, best_epoch=best_epoch, device=str(device))
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    model_name = NAMES[a.model].format(rnn=a.rnn.upper())
    if a.model != "textcnn":
        model_name += " + Attention" if not a.no_attention else " (max-pool)"
    rep = (f"GloVe {a.emb_dim}d" if a.embed == "glove" else "Random embeddings") + (" (frozen)" if a.freeze else "")
    append_result(a.results, {
        "Model": model_name, "Representation": rep,
        "Val Macro-F1": round(best_f1, 4), "Test Macro-F1": round(test_f1, 4),
        "Test Accuracy": round(test_acc, 4), "Test Weighted-F1": round(test_wf1, 4),
        "Job title": "removed" if (a.remove_title or getattr(a, "title_removed_by_column", False)) else "kept", "Max len": a.max_len, "Truncation": a.truncate,
        "Params": n_params, "Train time (s)": round(train_time, 1), "Best epoch": best_epoch,
        "Seed": a.seed, "Run": name,
    })
    print(f"saved to {out}/ and appended to {a.results}")
    return test_f1


def append_result(path, row):
    """Append one row; if the file already exists with other columns (e.g. the baseline
    script's), merge the columns instead of writing misaligned rows."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        old = pd.read_csv(path)
        if list(old.columns) != list(row):
            pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
            return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def plot_curves(history, path, title):
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="validation")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1"); ax[1].plot(h.epoch, h.val_acc, label="val accuracy")
    ax[1].set_title("Validation"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(cm, labels, path, title):
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=0.04); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
