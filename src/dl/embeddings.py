"""Random vs. pre-trained (GloVe) embedding matrices."""
import numpy as np


def random_matrix(vocab_size, dim, seed=42):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.1, (vocab_size, dim)).astype("float32")
    m[0] = 0.0  # padding
    return m


def glove_matrix(path, vocab, dim=300, seed=42):
    """Rows for words found in GloVe get the GloVe vector; the rest stay random.
    Returns (matrix, coverage) — report the coverage in the Methodology section."""
    m = random_matrix(len(vocab), dim, seed)
    found = 0
    with open(path, encoding="utf8") as f:
        for line in f:
            word, *vals = line.rstrip().split(" ")
            idx = vocab.stoi.get(word)
            if idx is not None and len(vals) == dim:
                m[idx] = np.asarray(vals, dtype="float32")
                found += 1
    m[0] = 0.0
    return m, found / max(len(vocab) - 2, 1)
