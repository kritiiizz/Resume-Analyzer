"""Audit text duplicates and preserve partition membership."""

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import groupby

from sklearn.model_selection import StratifiedGroupKFold

try:
    from .preprocess import TOKEN, canonical
except ImportError:
    from preprocess import TOKEN, canonical


PARTITIONS = ("train", "val", "test")
NEAR_THRESHOLD = 0.85
SHINGLE_SIZE = 5
MIN_SHINGLES = 25


def exact_duplicates(rows):
    by_text = defaultdict(list)
    for row in rows:
        key = canonical(row["Resume_str"])
        if key:
            by_text[key].append(row["ID"])
    return [ids for ids in by_text.values() if len(ids) > 1]


def near_duplicates(rows):
    """Find all eligible Jaccard pairs using a global rare-first prefix index."""
    documents = {}
    for row in rows:
        words = TOKEN.findall(canonical(row["Resume_str"]))
        shingles = {
            " ".join(words[start:start + SHINGLE_SIZE])
            for start in range(len(words) - SHINGLE_SIZE + 1)
        }
        if len(shingles) >= MIN_SHINGLES:
            documents[row["ID"]] = shingles

    document_frequency = Counter(shingle for items in documents.values() for shingle in items)
    index = defaultdict(list)
    pairs = []
    for identity in sorted(documents, key=lambda key: (len(documents[key]), key)):
        shingles = documents[identity]
        minimum = math.ceil(NEAR_THRESHOLD * len(shingles))
        ordered = sorted(shingles, key=lambda item: (document_frequency[item], item))
        # A qualifying pair shares at least one rare-first prefix term.
        prefix = ordered[:len(shingles) - minimum + 1]
        candidates = {other for item in prefix for other in index[item]}
        for other in sorted(candidates):
            previous = documents[other]
            if len(previous) < minimum:
                continue
            shared = len(shingles.intersection(previous))
            if not shared:
                continue
            score = shared / (len(shingles) + len(previous) - shared)
            if score >= NEAR_THRESHOLD:
                pairs.append((other, identity, score))
        for item in prefix:
            index[item].append(identity)
    return pairs


@dataclass
class Split:
    mode: str
    partitions: dict
    excluded: dict
    provided_text_rows: int = 0
    provided_text_mismatches: int = 0

    def by_id(self):
        return {
            **{identity: "excluded" for identity in self.excluded},
            **{identity: name for name in PARTITIONS for identity in self.partitions[name]},
        }


def provisional_split(rows, exact_groups, near_pairs, seed):
    categories = {row["ID"]: row["Category"] for row in rows}
    excluded = {
        row["ID"]: "blank_resume" for row in rows if not canonical(row["Resume_str"])
    }
    parent = {identity: identity for identity in categories}

    def root(identity):
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    def join(first, second):
        parent[root(second)] = root(first)

    for ids in exact_groups:
        for key, members in groupby(
            sorted(ids, key=lambda i: (categories[i], i)), key=categories.get
        ):
            same_label = list(members)
            for identity in same_label[1:]:
                excluded[identity] = "exact_duplicate_same_label"
        for duplicate in ids[1:]:
            join(ids[0], duplicate)
    for first, second, _ in near_pairs:
        join(first, second)

    active = sorted(set(categories).difference(excluded))
    labels = [categories[identity] for identity in active]
    groups = [root(identity) for identity in active]
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = splitter.split(active, labels, groups)
    _, val_indices = next(folds)
    _, test_indices = next(folds)
    holdout = set(val_indices).union(test_indices)
    partitions = {
        "train": [identity for index, identity in enumerate(active) if index not in holdout],
        "val": [active[index] for index in val_indices],
        "test": [active[index] for index in test_indices],
    }
    return Split("provisional", partitions, excluded)


def normalized_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def existing_split(rows, directory):
    """Resolve a supplied ID or an unambiguous exact text/label pair."""
    master = {row["ID"]: row for row in rows}
    by_text = defaultdict(list)
    for row in rows:
        by_text[(row["Category"], normalized_newlines(row["Resume_str"]))].append(row["ID"])

    visited = set()
    partitions = {}
    provided_text_rows = provided_text_mismatches = 0
    for name in PARTITIONS:
        path = directory / f"{name}.csv"
        if not path.is_file():
            path = directory / f"task_a_{name}.csv"
        identities = []
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or ())
            if "Category" not in fields or not ({"ID", "Resume_str"} & fields):
                raise ValueError(f"{path.name} requires Category and either ID or Resume_str")
            for row in reader:
                location = f"{path.name}, CSV line {reader.line_num}"
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{location} has an invalid number of columns")
                identity = row.get("ID", "").strip()
                source_text = row.get("Resume_str")
                label = row["Category"].strip()
                if not identity and source_text is not None:
                    matches = by_text[(label, normalized_newlines(source_text))]
                    if len(matches) != 1:
                        raise ValueError(f"{location} needs an ID ({len(matches)} exact text matches)")
                    identity = matches[0]
                if identity not in master:
                    raise ValueError(f"{location} has an ID absent from the archive")
                if master[identity]["Category"] != label:
                    raise ValueError(f"{location} has a label that disagrees with the source")
                if identity in visited:
                    raise ValueError(f"{location} repeats an ID already included in the split")
                if source_text is not None:
                    provided_text_rows += 1
                    provided_text_mismatches += (
                        normalized_newlines(source_text) != normalized_newlines(master[identity]["Resume_str"])
                    )
                visited.add(identity)
                identities.append(identity)
        if not identities:
            raise ValueError(f"{path.name} has no matched records")
        partitions[name] = identities

    excluded = {identity: "not_in_supplied_split" for identity in master if identity not in visited}
    return Split("supplied", partitions, excluded, provided_text_rows, provided_text_mismatches)


def spans_partitions(identities, membership):
    return len({membership[identity] for identity in identities}.intersection(PARTITIONS)) > 1
