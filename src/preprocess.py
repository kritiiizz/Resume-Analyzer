"""Deterministic, label-independent text cleaning for CV classifiers.

The default pipeline deliberately keeps skill punctuation (C++, C#, .NET).
Optional lemmatization uses spaCy and requires the en_core_web_sm model.
"""
from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.I)
URL = re.compile(r"\b(?:https?://|www\.)[^\s<>]+", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d()\s.-]{7,}\d)(?!\w)")
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
WHITESPACE = re.compile(r"[ \t\f\v]+")
# The default scikit-learn token_pattern drops signs from C++ and C#.
SKILL_TOKEN_PATTERN = r"(?u)(?<!\w)[A-Za-z0-9][A-Za-z0-9_+#./-]*"

# Conservative lexical substitutions; optional because their effects should be ablated.
SKILL_ALIASES = (
    (re.compile(r"(?<!\w)power[\s-]*bi(?!\w)", re.I), "power_bi"),
    (re.compile(r"(?<!\w)node[.]js(?!\w)", re.I), "node.js"),
    (re.compile(r"(?<!\w)machine learning(?!\w)", re.I), "machine_learning"),
)


def clean_text(
    value: str,
    *,
    mask_pii: bool = True,
    lowercase: bool = True,
    normalize_skill_synonyms: bool = False,
    remove_title_line: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError("Resume text must be a string")
    text = unicodedata.normalize("NFKC", html.unescape(value))
    text = CONTROL.sub(" ", text).replace("\r\n", "\n").replace("\r", "\n")
    if remove_title_line:
        # A deliberately conservative heuristic, independent of the label.
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) >= 2 and 1 <= len(lines[0].split()) <= 5:
            if not re.search(r"\b(summary|profile|experience|education|skills)\b", lines[0], re.I):
                lines = lines[1:]
        text = "\n".join(lines)
    if mask_pii:
        text = EMAIL.sub(" piiemail ", text)
        text = URL.sub(" piiurl ", text)
        text = PHONE.sub(" piiphone ", text)
    if normalize_skill_synonyms:
        for pattern, replacement in SKILL_ALIASES:
            text = pattern.sub(replacement, text)
    text = "\n".join(WHITESPACE.sub(" ", line).strip() for line in text.splitlines())
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text.lower() if lowercase else text


class ResumeCleaner(BaseEstimator, TransformerMixin):
    """Pickle-safe transformer: cleaning is always applied at inference too."""

    def __init__(
        self,
        mask_pii: bool = True,
        lowercase: bool = True,
        normalize_skill_synonyms: bool = False,
        remove_title_line: bool = False,
        lemmatize: bool = False,
    ):
        self.mask_pii = mask_pii
        self.lowercase = lowercase
        self.normalize_skill_synonyms = normalize_skill_synonyms
        self.remove_title_line = remove_title_line
        self.lemmatize = lemmatize

    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]) -> list[str]:
        cleaned = [
            clean_text(
                text,
                mask_pii=self.mask_pii,
                lowercase=self.lowercase,
                normalize_skill_synonyms=self.normalize_skill_synonyms,
                remove_title_line=self.remove_title_line,
            )
            for text in X
        ]
        if not self.lemmatize:
            return cleaned
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Optional lemmatization requires spaCy and en_core_web_sm; "
                "install spaCy and run: python -m spacy download en_core_web_sm"
            ) from exc
        return [
            " ".join(token.lemma_ for token in doc if not token.is_space)
            for doc in nlp.pipe(cleaned, batch_size=64)
        ]
