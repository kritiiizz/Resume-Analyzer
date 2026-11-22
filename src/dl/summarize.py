"""Turn reports/results.csv into report-ready tables and a comparison chart.

    python -m src.dl.summarize
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLS = ["Model", "Representation", "Val Macro-F1", "Test Macro-F1", "Test Accuracy"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="reports/results.csv")
    p.add_argument("--out", default="reports")
    a = p.parse_args(argv)
    df = pd.read_csv(a.results)
    if "Job title" not in df.columns:  # rows from the baseline scripts may not have it
        df["Job title"] = "kept"
    df["Job title"] = df["Job title"].fillna("kept")
    df["Representation"] = df.get("Representation", pd.Series(index=df.index, dtype=object)).fillna("-")
    for c in ("Max len", "Truncation", "Seed"):
        if c not in df.columns:
            df[c] = None

    default = ((df["Max len"].fillna(512) == 512) & (df["Truncation"].fillna("head") == "head")
               & (df["Seed"].fillna(42) == 42))
    main_rows = df[default & (df["Job title"] == "kept")]
    main_tab = main_rows[COLS].drop_duplicates(subset=["Model", "Representation"], keep="last") \
        .sort_values("Test Macro-F1", ascending=False)
    main_tab.to_csv(f"{a.out}/table_main_comparison.csv", index=False)
    print("\nMain comparison (job title kept, test set)\n" + main_tab.to_string(index=False))

    leak = df[default].pivot_table(index=["Model", "Representation"], columns="Job title", values="Test Macro-F1", aggfunc="last")
    if {"kept", "removed"} <= set(leak.columns):
        leak = leak.dropna().assign(drop=lambda t: (t["kept"] - t["removed"]).round(4))
        leak.to_csv(f"{a.out}/table_title_leakage.csv")
        print("\nLabel leakage (Test Macro-F1)\n" + leak.to_string())

    if "Seed" in df.columns and df["Seed"].nunique() > 1:
        stab = df.groupby(["Model", "Representation", "Job title", "Max len"])["Test Macro-F1"].agg(["mean", "std", "count"])
        stab = stab[stab["count"] > 1].round(4)
        if len(stab):
            stab.to_csv(f"{a.out}/table_seed_stability.csv")
            print("\nSeed stability\n" + stab.to_string())

    t = main_tab.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.2))
    ax.barh(t["Model"] + " | " + t["Representation"].astype(str), t["Test Macro-F1"], color="#4a6f94")
    for i, v in enumerate(t["Test Macro-F1"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Test macro-F1"); ax.set_xlim(0, min(1, t["Test Macro-F1"].max() + 0.12))
    ax.set_title("Model comparison (resume category, 24 classes)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{a.out}/model_comparison.png", dpi=160); plt.close(fig)
    print(f"\nsaved tables and {a.out}/model_comparison.png")


if __name__ == "__main__":
    main()
