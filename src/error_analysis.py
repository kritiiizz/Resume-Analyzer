"""Error analysis for any model's test predictions.

Input: a CSV with columns  true, pred  and optionally  confidence, top3, text (or text_start), ID.
       (src/dl/train.py writes runs/<run>/test_predictions.csv in this format.)

    python -m src.error_analysis --pred runs/<run>/test_predictions.csv --name bilstm_glove
    python -m src.error_analysis --pred a.csv --compare b.csv --name bilstm_vs_distilbert

Writes to reports/error_analysis/<name>/:
  summary.md             numbers and tables to paste into the report
  per_class.csv          precision / recall / F1 / support per category
  confused_pairs.csv     most frequent (true -> predicted) confusions
  errors_to_label.csv    every wrong prediction with automatic tags + empty columns for manual labelling
  per_class_f1.png, confused_pairs.png, error_types.png
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

SHORT_WORDS = 150   # resumes shorter than this are tagged "short text"
LOW_CONF = 0.40     # predictions below this confidence are tagged "low confidence"


def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def auto_tags(row, labels, text_col):
    tags = []
    if "top3" in row and isinstance(row["top3"], str):
        top = [re.sub(r"\s-?[\d.]+$", "", p.strip()) for p in row["top3"].split(";")]
        tags.append("true label in top-3" if row["true"] in top else "true label not in top-3")
    if "confidence" in row and pd.notna(row["confidence"]):
        tags.append("low confidence" if row["confidence"] < LOW_CONF else "confident but wrong")
    if text_col:
        text = str(row[text_col])
        n = row["n_words"] if "n_words" in row and pd.notna(row.get("n_words")) else (len(text.split()) if text_col == "text" else None)
        if n is not None and n < SHORT_WORDS:
            tags.append("short text")
        head = norm(text[:120])
        if norm(row["pred"]) and norm(row["pred"]) in head:
            tags.append("title names predicted class")
        elif norm(row["true"]) and norm(row["true"]) in head:
            tags.append("title names true class")
    return "; ".join(tags)


def barh(series, title, path, xlabel, color="#4a6f94"):
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(series) + 1.2))
    ax.barh(series.index[::-1], series.values[::-1], color=color)
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10); ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def analyse(df, name, out):
    os.makedirs(out, exist_ok=True)
    labels = sorted(set(df["true"]) | set(df["pred"]))
    text_col = "text" if "text" in df.columns else ("text_start" if "text_start" in df.columns else None)
    acc = accuracy_score(df["true"], df["pred"])
    mf1 = f1_score(df["true"], df["pred"], average="macro", zero_division=0)

    p, r, f, s = precision_recall_fscore_support(df["true"], df["pred"], labels=labels, zero_division=0)
    per_class = pd.DataFrame({"category": labels, "precision": p.round(3), "recall": r.round(3),
                              "f1": f.round(3), "support": s}).sort_values("f1")
    per_class.to_csv(f"{out}/per_class.csv", index=False)
    barh(per_class.set_index("category")["f1"].sort_values(ascending=False), f"{name}: F1 per category (test)",
         f"{out}/per_class_f1.png", "F1")

    err = df[df["true"] != df["pred"]].copy()
    pairs = err.groupby(["true", "pred"]).size().sort_values(ascending=False).rename("count").reset_index()
    pairs["share of that true class"] = [round(c / (df["true"] == t).sum(), 2) for t, c in zip(pairs["true"], pairs["count"])]
    pairs.to_csv(f"{out}/confused_pairs.csv", index=False)
    top = pairs.head(15)
    if len(top):
        barh(pd.Series(top["count"].values, index=top["true"] + "  ->  " + top["pred"]),
             f"{name}: most frequent confusions (true -> predicted)", f"{out}/confused_pairs.png", "test resumes", "#a8553a")

    tag_counts = pd.Series(dtype=int)
    if len(err):
        err["auto_tags"] = err.apply(lambda row: auto_tags(row, labels, text_col), axis=1)
        tag_counts = err["auto_tags"].str.split("; ").explode().replace("", np.nan).dropna().value_counts()
        if len(tag_counts):
            barh(tag_counts, f"{name}: automatic error tags ({len(err)} errors)", f"{out}/error_types.png", "errors", "#6b7f5e")
    err["error_type (fill in)"] = ""
    err["notes (fill in)"] = ""
    sort_cols = [c for c in ("confidence",) if c in err.columns]
    (err.sort_values(sort_cols, ascending=False) if sort_cols else err).to_csv(f"{out}/errors_to_label.csv", index=False)

    lines = [f"# Error analysis: {name}", "",
             f"- Test resumes: {len(df)}; wrong predictions: {len(err)} ({len(err) / len(df):.1%})",
             f"- Accuracy {acc:.4f}, macro-F1 {mf1:.4f}"]
    if "top3" in df.columns and len(err):
        hit = err["auto_tags"].str.contains("true label in top-3").mean()
        lines.append(f"- For {hit:.0%} of the errors the true category was still in the model's top 3")
    lines += ["", "## Weakest categories", "", per_class.head(6).to_markdown(index=False), "",
              "## Strongest categories", "", per_class.tail(5).iloc[::-1].to_markdown(index=False), "",
              "## Most frequent confusions", "", pairs.head(12).to_markdown(index=False) if len(pairs) else "none", ""]
    if len(tag_counts):
        lines += ["## Automatic error tags (one error can have several)", "", tag_counts.rename("errors").to_markdown(), ""]
    lines += ["## Manual step", "",
              "Open `errors_to_label.csv`, read 30-50 errors and fill in `error_type` using a fixed list, e.g.:",
              "overlapping categories / mixed-career resume / label looks wrong / too short or noisy / generic wording / other.",
              "Then count the types and describe 3 examples in the report.", ""]
    open(f"{out}/summary.md", "w").write("\n".join(lines))
    return err


def compare(a, b, name_a, name_b, out):
    """Which errors are shared between two models and which are unique to one."""
    key = "ID" if "ID" in a.columns and "ID" in b.columns else ("row" if "row" in a.columns and "row" in b.columns else None)
    if key is None or len(a) != len(b):
        return "\nCould not compare: the two files need the same test rows and an `ID` or `row` column.\n"
    a = a.assign(**{key: a[key].astype(str)}); b = b.assign(**{key: b[key].astype(str)})
    m = a[[key, "true", "pred"]].merge(b[[key, "pred"]], on=key, suffixes=("_a", "_b"))
    wa, wb = m["pred_a"] != m["true"], m["pred_b"] != m["true"]
    rows = [("both wrong", int((wa & wb).sum())), (f"only {name_a} wrong", int((wa & ~wb).sum())),
            (f"only {name_b} wrong", int((~wa & wb).sum())), ("both correct", int((~wa & ~wb).sum()))]
    m[wa & wb].to_csv(f"{out}/errors_shared_by_both.csv", index=False)
    return "\n## Comparison of the two models\n\n" + pd.DataFrame(rows, columns=["case", "test resumes"]).to_markdown(index=False) + \
        "\n\nErrors both models make are the hard cases (see `errors_shared_by_both.csv`).\n"


def add_scores(df):
    """Derive top3 / confidence from per-class columns (prob_<label> or decision_score_<label>) if present."""
    for prefix, is_prob in (("prob_", True), ("decision_score_", False)):
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols and "top3" not in df.columns:
            labels = [c[len(prefix):] for c in cols]
            m = df[cols].to_numpy(dtype=float)
            order = np.argsort(-m, axis=1)[:, :3]
            df["top3"] = ["; ".join(f"{labels[j]} {m[r, j]:.2f}" for j in order[r]) for r in range(len(df))]
            if is_prob and "confidence" not in df.columns:
                df["confidence"] = m.max(1).round(4)
            df = df.drop(columns=cols)
    return df


def add_texts(df, texts):
    """Attach resume text by matching the prediction ID to an ID or sample_id column of the split file."""
    key = next((k for k in ("sample_id", "ID") if k in texts.columns and df["ID"].astype(str).isin(texts[k].astype(str)).all()), None)
    if key is None:
        print("warning: could not match prediction IDs to the text file; continuing without text")
        return df
    t = texts.rename(columns={"ID": "kaggle_ID"} if key == "sample_id" else {}).copy()
    t[key] = t[key].astype(str)
    df = df.assign(ID=df["ID"].astype(str)).merge(t[[key] + [c for c in ("kaggle_ID",) if c in t] + ["Resume_str"]],
                                                   left_on="ID", right_on=key, how="left")
    df["text"] = df.pop("Resume_str")
    df["n_words"] = df["text"].astype(str).str.split().str.len()
    if key != "ID":
        df = df.drop(columns=[key])
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--compare", default=None, help="second predictions CSV on the same test rows")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="reports/error_analysis")
    ap.add_argument("--texts", default=None, help="split CSV with the resume text (Resume_str) to attach to each prediction")
    a = ap.parse_args(argv)
    name = a.name or os.path.basename(os.path.dirname(os.path.abspath(a.pred)))
    out = os.path.join(a.out, name)
    df = pd.read_csv(a.pred)
    missing = {"true", "pred"} - set(df.columns)
    if missing:
        raise SystemExit(f"{a.pred} needs columns true and pred; found {list(df.columns)}")
    df = add_scores(df)
    if a.texts:
        df = add_texts(df, pd.read_csv(a.texts))
    err = analyse(df, name, out)
    if a.compare:
        other = pd.read_csv(a.compare)
        with open(f"{out}/summary.md", "a") as f:
            f.write(compare(df, other, name.split("_vs_")[0] if "_vs_" in name else "first model",
                            name.split("_vs_")[1] if "_vs_" in name else "second model", out))
    print(open(f"{out}/summary.md").read())
    print(f"saved to {out}/  ({len(err)} errors to label)")


if __name__ == "__main__":
    main()
