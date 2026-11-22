"""Load a trained run for the Streamlit demo.

    from src.dl.predict import load_run
    clf = load_run("runs/bilstm_lstm_att_glove_title_len512_s42")
    clf.predict(resume_text)          # -> [("INFORMATION-TECHNOLOGY", 0.81), ("ENGINEERING", 0.07), ...]
    clf.top_words(resume_text, k=15)  # -> words with the highest attention (attention models only)
"""
import json
import os

import numpy as np
import torch

from .data import Vocab
from .models import build_model
from .text import prepare


class Predictor:
    def __init__(self, run_dir, device="cpu"):
        with open(os.path.join(run_dir, "config.json")) as f:
            self.cfg = json.load(f)
        with open(os.path.join(run_dir, "vocab.json")) as f:
            self.vocab = Vocab(json.load(f))
        self.labels = self.cfg["labels"]
        self.device = torch.device(device)
        dummy = np.zeros((self.cfg["vocab_size"], self.cfg["emb_dim"]), dtype="float32")
        self.model = build_model(self.cfg["model"], dummy, len(self.labels), self.cfg["model_cfg"], pack=True)
        self.model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=self.device))
        self.model.to(self.device).eval()

    def _encode(self, text):
        tokens = prepare(text, self.cfg["remove_title"], not self.cfg["no_clean"])
        ids = self.vocab.encode(tokens, self.cfg["max_len"], self.cfg["truncate"])
        ids = ids + [0] * max(0, 5 - len(ids))
        x = torch.tensor([ids], device=self.device)
        return x, torch.tensor([len(ids)])

    @torch.no_grad()
    def predict(self, text, k=3):
        x, lengths = self._encode(text)
        p = torch.softmax(self.model(x, lengths), -1)[0].cpu().numpy()
        return [(self.labels[i], float(p[i])) for i in np.argsort(-p)[:k]]

    @torch.no_grad()
    def top_words(self, text, k=15):
        x, lengths = self._encode(text)
        _, attn = self.model(x, lengths, return_attn=True)
        if attn is None:
            return []
        weights = attn[0].cpu().numpy()
        words = [self.vocab.itos[i] for i in x[0].tolist()]
        scores = {}
        for w, s in zip(words, weights):
            if w not in ("<pad>", "<unk>"):
                scores[w] = scores.get(w, 0.0) + float(s)
        return sorted(scores.items(), key=lambda t: -t[1])[:k]


def load_run(run_dir, device="cpu"):
    return Predictor(run_dir, device)
