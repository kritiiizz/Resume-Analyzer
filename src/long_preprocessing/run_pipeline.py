"""Prepare and audit the Task A resume data from the source archive or CSV."""

import argparse
import csv
import hashlib
import io
import json
import tempfile
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

try:
    from .preprocess import (
        CONTACT_TOKENS, LEMMA_COLUMNS, TEXT_COLUMNS, add_lemmatized_text, label_phrase_present,
        preprocess, remove_header,
    )
    from .splits import (
        MIN_SHINGLES, NEAR_THRESHOLD, PARTITIONS, SHINGLE_SIZE, exact_duplicates, existing_split,
        near_duplicates, provisional_split, spans_partitions,
    )
except ImportError:
    from preprocess import (
        CONTACT_TOKENS, LEMMA_COLUMNS, TEXT_COLUMNS, add_lemmatized_text, label_phrase_present,
        preprocess, remove_header,
    )
    from splits import (
        MIN_SHINGLES, NEAR_THRESHOLD, PARTITIONS, SHINGLE_SIZE, exact_duplicates, existing_split,
        near_duplicates, provisional_split, spans_partitions,
    )


SOURCE_COLUMNS = {"ID", "Resume_str", "Resume_html", "Category"}
SOURCE_MEMBER = "Resume/Resume.csv"
MANIFEST_COLUMNS = ("ID", "Category", "split", "exclusion_reason", "header_status")
HOLDOUT_TERM_MIN_DOCS = 5
HANDOFF_TEXT = {
    "raw_no_header.csv": "text_raw_no_header",
    "light.csv": "text_light_no_header",
    "classical.csv": "text_classical_no_header",
    "stemmed.csv": "text_stemmed_no_header",
    "lemmatized.csv": "text_lemmatized_no_header",
}


def load_csv(file):
    reader = csv.DictReader(file)
    if len(reader.fieldnames or ()) != 4 or set(reader.fieldnames) != SOURCE_COLUMNS:
        raise ValueError("Expected CSV columns ID, Resume_str, Resume_html and Category")
    rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError("The CSV is empty or has a malformed record")
    ids = [row["ID"] for row in rows]
    if len(set(ids)) != len(ids) or not all(i and i == i.strip() for i in ids):
        raise ValueError("IDs must be nonempty, unpadded and unique")
    if not all(row["Category"] and row["Category"] == row["Category"].strip() for row in rows):
        raise ValueError("Categories must be nonempty and unpadded")
    return rows


def load_source(path):
    with path.open("rb") as source:
        fingerprint = hashlib.file_digest(source, "sha256").hexdigest()
    info = {"filename": path.name, "sha256": fingerprint, "pdf_count": None,
            "pdf_keys_match_csv": None, "archive_crc_ok": None, "csv_member": None}

    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as source:
            return load_csv(source), info

    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("A ZIP member failed its CRC check")
        with archive.open(SOURCE_MEMBER) as raw:
            rows = load_csv(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        pdfs = [PurePosixPath(name) for name in archive.namelist() if name.lower().endswith(".pdf")]
        pdf_keys = {(name.stem, name.parent.name.casefold()) for name in pdfs}
        csv_keys = {(row["ID"], row["Category"].casefold()) for row in rows}
        info.update(pdf_count=len(pdfs), archive_crc_ok=True, csv_member=SOURCE_MEMBER,
                    pdf_keys_match_csv=(len(pdfs) == len(pdf_keys) and pdf_keys == csv_keys))
        return rows, info


def write_csv(path, records, fields):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_handoff(path, rows, index, split):
    processed = path / "processed"
    rosters = path / "split"
    processed.mkdir(parents=True)
    rosters.mkdir()
    for name, column in HANDOFF_TEXT.items():
        if column in rows[0]:
            records = ({**row, "text": row[column]} for row in rows)
            write_csv(processed / name, records, ("ID", "Category", "split", "text"))
    for name, identities in split.partitions.items():
        write_csv(rosters / f"{name}.csv", (index[identity] for identity in identities), ("ID", "Category"))


def distribution(values):
    return {
        "min": int(min(values)), "median": round(float(np.median(values)), 1),
        "p95": round(float(np.percentile(values, 95)), 1), "max": int(max(values)),
    }


def prepare(rows, split, lemmatize):
    membership = split.by_id()
    lengths = defaultdict(list)
    masks_by_type = Counter()
    status_by_label = defaultdict(Counter)
    terms_by_label = defaultdict(Counter)
    variant_lengths = defaultdict(list)
    for row in rows:
        raw = row["Resume_str"]
        label = row["Category"]
        header = remove_header(raw, row["Resume_html"])
        fields, masks = preprocess(raw, header)
        row.update(fields)
        row["split"] = membership[row["ID"]]
        row["exclusion_reason"] = split.excluded.get(row["ID"], "")
        row["header_status"] = header.status
        row["literal_label_in_removed_header"] = bool(
            header.status == "removed" and label_phrase_present(header.header, label)
        )
        row["literal_label_in_body"] = label_phrase_present(header.body, label)

        lengths[label].append(len(raw.split()))
        status_by_label[label][header.status] += 1
        status_by_label[label]["literal_header_label"] += row["literal_label_in_removed_header"]
        for key, count in masks.items():
            masks_by_type[f"{key}_matches"] += count
            masks_by_type[f"{key}_records"] += (count > 0)
        if row["split"] == "train":
            tokens = {
                token for token in row["text_classical_no_header"].split()
                if token not in CONTACT_TOKENS and len(token) >= 3 and token.isalpha()
            }
            terms_by_label[label].update(tokens)

    lemma_model = add_lemmatized_text(rows) if lemmatize else None
    text_columns = TEXT_COLUMNS + (LEMMA_COLUMNS if lemmatize else ())
    for row in rows:
        for column in text_columns:
            variant_lengths[column].append(len(row[column].split()))

    count_by_label_split = Counter((row["Category"], row["split"]) for row in rows)
    category_stats = []
    top_terms = []
    for label, words in sorted(lengths.items()):
        counts = status_by_label[label]
        category_stats.append({
            "category": label, "records": len(words), **distribution(words),
            **{name: count_by_label_split[label, name] for name in (*PARTITIONS, "excluded")},
            "headers_removed": counts["removed"], "literal_header_label": counts["literal_header_label"],
        })
        candidates = sorted(terms_by_label[label].items(), key=lambda item: (-item[1], item[0]))
        top_terms.extend(
            {"category": label, "term": term, "train_documents": count}
            for term, count in candidates[:5] if count >= HOLDOUT_TERM_MIN_DOCS
        )

    details = {
        "raw_word_lengths_by_category": lengths,
        "contact_patterns": dict(sorted(masks_by_type.items())),
        "header_status": dict(sorted(Counter(row["header_status"] for row in rows).items())),
        "literal_label_in_removed_header": sum(row["literal_label_in_removed_header"] for row in rows),
        "literal_label_in_body_when_header_removed": sum(
            row["literal_label_in_body"] and row["header_status"] == "removed" for row in rows
        ),
        "whitespace_word_lengths": {
            column: distribution(values) for column, values in variant_lengths.items()
        },
        "lemma_model": lemma_model,
    }
    return text_columns, category_stats, top_terms, details


def verify(rows, partitions, exact, near, text_columns, source):
    expected_ids = {row["ID"] for row in rows}
    membership = partitions.by_id()
    active_lists = [identity for split in PARTITIONS for identity in partitions.partitions[split]]
    active_labels = {
        (row["split"], row["Category"]) for row in rows if row["split"] in PARTITIONS
    }
    all_labels = {row["Category"] for row in rows}
    exact_crossings = [ids for ids in exact if spans_partitions(ids, membership)]
    near_crossings = [pair for pair in near if spans_partitions(pair[:2], membership)]
    qa = {
        "all_source_ids_accounted_for": set(membership) == expected_ids,
        "no_id_crossing_or_active_excluded_overlap": (
            len(set(active_lists)) == len(active_lists)
            and not set(active_lists).intersection(partitions.excluded)
        ),
        "membership_matches_every_row": all(row["split"] == membership[row["ID"]] for row in rows),
        "all_categories_in_each_partition": len(active_labels) == 3 * len(all_labels),
        "headers_remove_only_a_prefix": all(
            row["Resume_str"].endswith(row["text_raw_no_header"])
            if row["header_status"] == "removed"
            else row["Resume_str"] == row["text_raw_no_header"]
            for row in rows
        ),
        "variants_present_in_every_row": all(
            all(isinstance(row.get(column), str) for column in text_columns) for row in rows
        ),
        "exact_duplicate_groups_across_partitions": len(exact_crossings),
        "near_duplicate_pairs_across_partitions": len(near_crossings),
        "archive_crc_ok": source["archive_crc_ok"],
        "accompanying_pdfs_match_csv": source["pdf_keys_match_csv"],
    }
    required = [
        "all_source_ids_accounted_for", "no_id_crossing_or_active_excluded_overlap",
        "membership_matches_every_row", "headers_remove_only_a_prefix", "variants_present_in_every_row",
    ]
    if partitions.mode == "provisional":
        required.append("all_categories_in_each_partition")
        if exact_crossings or near_crossings:
            raise AssertionError("The provisional split crosses a detected duplicate group")
    if not all(qa[key] for key in required):
        raise AssertionError(f"Dataset integrity check failed: {qa}")
    return qa


def run(source_path, target, supplied_splits, seed, lemmatize, handoff=False):
    csv.field_size_limit(10 * 1024 * 1024)
    if target.exists():
        raise FileExistsError(f"Choose a new output directory: {target}")
    print("Reading and validating the source archive.", flush=True)
    rows, source = load_source(source_path)
    canonical_groups = exact_duplicates(rows)
    all_similar = near_duplicates(rows)
    exact_group_index = {
        identity: index for index, group in enumerate(canonical_groups) for identity in group
    }
    near = [
        pair for pair in all_similar
        if pair[0] not in exact_group_index
        or exact_group_index[pair[0]] != exact_group_index.get(pair[1])
    ]
    labels_by_id = {row["ID"]: row["Category"] for row in rows}
    split = (existing_split(rows, supplied_splits) if supplied_splits
             else provisional_split(rows, canonical_groups, near, seed))
    print(f"Preparing aligned text variants for {len(rows)} resumes.", flush=True)
    columns, stats, top_terms, details = prepare(rows, split, lemmatize)
    qa = verify(rows, split, canonical_groups, near, columns, source)
    summary = {
        "source": source, "records": len(rows), "categories": len(stats),
        "blank_records": sum(not row["Resume_str"].strip() for row in rows),
        "no_line_break_records": sum(
            "\n" not in row["Resume_str"] and "\r" not in row["Resume_str"] for row in rows
        ),
        "exact_duplicate_groups": len(canonical_groups),
        "exact_duplicate_extra_records": sum(len(group) - 1 for group in canonical_groups),
        "exact_duplicate_label_conflicts": sum(
            len({labels_by_id[identity] for identity in group}) > 1 for group in canonical_groups
        ),
        "near_duplicate_candidates": len(near),
        "near_duplicate_rule": {"shingle_size": SHINGLE_SIZE, "minimum_shingles": MIN_SHINGLES,
                                "jaccard_threshold": NEAR_THRESHOLD},
        "split_mode": split.mode,
        "seed": seed if split.mode == "provisional" else None,
        "split_counts": {
            **{name: len(split.partitions[name]) for name in PARTITIONS},
            "excluded": len(split.excluded),
        },
        "exclusion_reasons": dict(Counter(split.excluded.values())),
        "provided_text_rows": split.provided_text_rows,
        "provided_text_mismatches": split.provided_text_mismatches,
        "versions": {package: version(package) for package in ("numpy", "scikit-learn", "nltk")},
        "integrity_checks": qa,
        **details,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    index = {row["ID"]: row for row in rows}
    with tempfile.TemporaryDirectory(prefix=".taska-", dir=target.parent) as temp:
        staged = Path(temp) / "result"
        if handoff:
            write_handoff(staged, rows, index, split)
        else:
            private = staged / "private"
            public = staged / "aggregate"
            private.mkdir(parents=True)
            public.mkdir()
            fields = (*MANIFEST_COLUMNS, "literal_label_in_removed_header", *columns)
            write_csv(private / "master.csv", rows, fields)
            write_csv(private / "split_manifest.csv", rows, MANIFEST_COLUMNS)
            for name, identities in split.partitions.items():
                write_csv(private / f"{name}.csv", (index[identity] for identity in identities), fields)
            write_json(private / "duplicate_review.json", {
                "exact_groups": canonical_groups,
                "near_pairs": [{"ids": [first, second], "jaccard": round(score, 6)}
                               for first, second, score in near],
            })
            write_csv(public / "category_stats.csv", stats, stats[0].keys())
            write_csv(public / "top_train_terms.csv", top_terms, ("category", "term", "train_documents"))
            write_json(public / "data_audit.json", summary)
        staged.rename(target)
    print(f"Integrated dataset checks passed. Partition sizes: {summary['split_counts']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Task A archive.zip or Resume.csv")
    parser.add_argument("--out", required=True, type=Path, help="A new output directory")
    parser.add_argument("--official-splits", type=Path, help="Directory containing train/val/test CSVs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--handoff", action="store_true", help="Write the processed and split folders from the handoff")
    parser.add_argument("--skip-lemmatization", action="store_true",
                        help="Prepare the other variants without the spaCy English model")
    args = parser.parse_args()
    run(args.source, args.out, args.official_splits, args.seed, not args.skip_lemmatization,
        handoff=args.handoff)


if __name__ == "__main__":
    main()
