"""Loading the splits, vocabulary, encoding and batching."""
import os
from collections import Counter

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

PAD, UNK = "<pad>", "<unk>"


def load_splits(path, text_col="Resume_str", label_col="Category", seed=42):
    """Return (train, val, test) DataFrames.

    path = a folder with train.csv / val.csv / test.csv (the shared team split — preferred), or
    path = the raw Kaggle Resume.csv, which is split here the same way the team agreed:
           duplicates dropped, stratified 80/10/10, random_state=42.
    """
    if os.path.isdir(path):
        parts = [pd.read_csv(os.path.join(path, f"{s}.csv")) for s in ("train", "val", "test")]
    else:
        df = pd.read_csv(path)
        df = df.dropna(subset=[text_col, label_col]).drop_duplicates(subset=text_col)
        train, temp = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=seed)
        val, test = train_test_split(temp, test_size=0.5, stratify=temp[label_col], random_state=seed)
        parts = [train, val, test]
    for p in parts:
        missing = {text_col, label_col} - set(p.columns)
        if missing:
            raise ValueError(f"Missing columns {missing}; use --text-col / --label-col. Found: {list(p.columns)}")
    return [p.reset_index(drop=True) for p in parts]


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq=2, max_size=30000):
        """Built from the TRAINING split only (no information from val/test)."""
        counts = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in counts.most_common() if c >= min_freq][: max_size - 2]
        return cls([PAD, UNK] + words)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len, mode="head"):
        ids = [self.stoi.get(t, 1) for t in tokens]
        if len(ids) > max_len:
            if mode == "head_tail":  # first quarter + last three quarters
                h = max_len // 4
                ids = ids[:h] + ids[-(max_len - h):]
            else:
                ids = ids[:max_len]
        return ids or [1]


class SeqDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids, self.labels = ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def collate(batch, min_len=5):
    """Pad to the longest sequence in the batch (at least min_len for the CNN kernels)."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    width = max(int(lengths.max()), min_len)
    x = torch.zeros(len(seqs), width, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    return x, lengths, torch.tensor(labels)
