"""The three architectures.

All models take (x, lengths) where x is a padded LongTensor [batch, seq] (pad id 0)
and return logits [batch, num_classes]. RNN models with attention can also return the
attention weights, which the demo uses to highlight words.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

NEG = -1e4


def make_embedding(weights, freeze=False):
    return nn.Embedding.from_pretrained(torch.as_tensor(weights), freeze=freeze, padding_idx=0)


class Attention(nn.Module):
    """Additive attention pooling over time steps (padding masked out)."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, h, mask):
        s = self.score(torch.tanh(self.proj(h))).squeeze(-1).masked_fill(~mask, NEG)
        a = torch.softmax(s, dim=-1)
        return (a.unsqueeze(-1) * h).sum(1), a


def masked_max(h, mask):
    return h.masked_fill(~mask.unsqueeze(-1), NEG).max(dim=1).values


class _RNNHead(nn.Module):
    """Shared: (Bi)LSTM/GRU over a sequence of features, then attention or max pooling."""

    def __init__(self, in_dim, hidden, layers, rnn, dropout, attention, pack):
        super().__init__()
        cls = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = cls(in_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.attn = Attention(2 * hidden) if attention else None
        self.pack = pack
        self.out_dim = 2 * hidden

    def forward(self, feats, lengths, mask):
        if self.pack:  # skips padding exactly; turned off on Apple MPS for compatibility
            packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = pad_packed_sequence(self.rnn(packed)[0], batch_first=True, total_length=feats.size(1))
        else:
            h, _ = self.rnn(feats)
        if self.attn is not None:
            return self.attn(h, mask)
        return masked_max(h, mask), None


class RNNClassifier(nn.Module):
    """Embedding -> 2-layer BiLSTM (or BiGRU) -> attention / max-pool -> dropout -> linear."""

    def __init__(self, weights, num_classes, rnn="lstm", hidden=128, layers=2, dropout=0.4,
                 attention=True, freeze=False, pack=True):
        super().__init__()
        self.emb = make_embedding(weights, freeze)
        self.drop = nn.Dropout(dropout)
        self.head = _RNNHead(self.emb.embedding_dim, hidden, layers, rnn, dropout, attention, pack)
        self.fc = nn.Linear(self.head.out_dim, num_classes)

    def forward(self, x, lengths, return_attn=False):
        mask = x != 0
        pooled, attn = self.head(self.drop(self.emb(x)), lengths, mask)
        logits = self.fc(self.drop(pooled))
        return (logits, attn) if return_attn else logits


class TextCNN(nn.Module):
    """Kim (2014): parallel convolutions (widths 3/4/5) -> max-over-time -> dropout -> linear."""

    def __init__(self, weights, num_classes, kernel_sizes=(3, 4, 5), filters=100, dropout=0.5, freeze=False):
        super().__init__()
        self.emb = make_embedding(weights, freeze)
        d = self.emb.embedding_dim
        self.convs = nn.ModuleList(nn.Conv1d(d, filters, k) for k in kernel_sizes)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernel_sizes), num_classes)

    def forward(self, x, lengths, return_attn=False):
        e = self.drop(self.emb(x)).transpose(1, 2)  # [B, d, T]
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(e))  # [B, F, T-k+1]
            valid = (x != 0)[:, : c.size(2)]  # mask windows that start in padding
            pooled.append(c.masked_fill(~valid.unsqueeze(1), NEG).max(dim=2).values)
        logits = self.fc(self.drop(torch.cat(pooled, dim=1)))
        return (logits, None) if return_attn else logits


class CNNBiLSTM(nn.Module):
    """Embedding -> Conv1d (local n-gram features, length kept) -> BiLSTM -> attention / max-pool -> linear."""

    def __init__(self, weights, num_classes, filters=128, kernel=3, hidden=128, layers=1, dropout=0.4,
                 attention=True, rnn="lstm", freeze=False, pack=True):
        super().__init__()
        self.emb = make_embedding(weights, freeze)
        self.conv = nn.Conv1d(self.emb.embedding_dim, filters, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)
        self.head = _RNNHead(filters, hidden, layers, rnn, dropout, attention, pack)
        self.fc = nn.Linear(self.head.out_dim, num_classes)

    def forward(self, x, lengths, return_attn=False):
        mask = x != 0
        c = torch.relu(self.conv(self.drop(self.emb(x)).transpose(1, 2))).transpose(1, 2)
        pooled, attn = self.head(self.drop(c), lengths, mask)
        logits = self.fc(self.drop(pooled))
        return (logits, attn) if return_attn else logits


def build_model(name, weights, num_classes, cfg, pack):
    common = dict(dropout=cfg["dropout"], freeze=cfg["freeze"])
    if name == "bilstm":
        return RNNClassifier(weights, num_classes, rnn=cfg["rnn"], hidden=cfg["hidden"], layers=cfg["layers"],
                             attention=cfg["attention"], pack=pack, **common)
    if name == "textcnn":
        return TextCNN(weights, num_classes, filters=cfg["filters"], **common)
    if name == "cnn_bilstm":
        return CNNBiLSTM(weights, num_classes, filters=cfg["filters"], hidden=cfg["hidden"], rnn=cfg["rnn"],
                         attention=cfg["attention"], pack=pack, **common)
    raise ValueError(f"unknown model {name}")
