"""Top-16 radiomic feature selection (train-only, plan section 3.4).

Produces three artifacts:
  data/processed/top16_features.json       -- flat 16-feature list for the
                                              paper-style splits.json train split
                                              (frozen; dataset reads this)
  data/processed/top16_features_folds.json -- per-fold lists for 5-fold CV
  data/processed/radiomics_top16.csv       -- radiomics_all.csv reduced to the
                                              paper-split 16 features (dataset input)

Selection uses ANOVA F-value (SelectKBest / f_classif) fit on training data only.
"""

import json
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

K = 16
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
META_COLS = {"patient_id", "slice_idx", "label"}


def feature_columns(df):
    return [c for c in df.columns if c not in META_COLS]


def select_top_k(X, y, k=K):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    return list(X.columns[selector.get_support()])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Select train-only top-16 radiomic features")
    parser.add_argument("--split-file", default=str(PROCESSED / "splits.json"),
                        help="Patient split JSON used to identify training patients")
    parser.add_argument("--output-suffix", default="",
                        help="Optional suffix for feature and reduced-radiomics artifacts")
    args = parser.parse_args(argv)

    if args.output_suffix and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.output_suffix):
        parser.error("--output-suffix may contain only letters, digits, dot, underscore, and hyphen")
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    feature_output = PROCESSED / f"top16_features{suffix}.json"
    folds_output = PROCESSED / f"top16_features_folds{suffix}.json"
    radiomics_output = PROCESSED / f"radiomics_top16{suffix}.csv"

    df = pd.read_csv(PROCESSED / "radiomics_all.csv")
    cols = feature_columns(df)

    with open(args.split_file) as f:
        splits = json.load(f)
    with open(PROCESSED / "cv_folds.json") as f:
        folds_dict = json.load(f)
    folds = [v for _, v in sorted(folds_dict.items(), key=lambda kv: kv[0])]

    # Paper-style split: select on the train patients only, freeze for val/test.
    train_df = df[df["patient_id"].isin(set(splits["train"]))]
    paper_selected = select_top_k(train_df[cols], train_df["label"].values)
    with open(feature_output, "w") as f:
        json.dump(paper_selected, f, indent=2)

    # 5-fold CV: one train-only selection per fold.
    per_fold = {}
    union = set()
    for i, fold in enumerate(folds):
        fold_train = df[df["patient_id"].isin(set(fold["train"]))]
        selected = select_top_k(fold_train[cols], fold_train["label"].values)
        per_fold[f"fold_{i}"] = selected
        union.update(selected)
    with open(folds_output, "w") as f:
        json.dump({"folds": per_fold, "union": sorted(union)}, f, indent=2)

    # Dataset input: radiomics_all reduced to the paper-split selection.
    keep = ["patient_id", "slice_idx", "label"] + paper_selected
    df[keep].to_csv(radiomics_output, index=False)

    print(f"paper-split top-16 -> {feature_output.name} ({len(paper_selected)} features)")
    for feat in paper_selected:
        print(f"  {feat}")
    print(f"per-fold top-16  -> {folds_output.name} (union {len(union)} features)")
    print(f"{radiomics_output.name} -> {len(keep)} columns")


if __name__ == "__main__":
    main()
