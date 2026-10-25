"""Create inner-train / early-stop validation / outer-test patient splits for CV."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe nested train/validation/test splits from cv_folds.json"
    )
    parser.add_argument("--folds-file", default=str(PROCESSED / "cv_folds.json"))
    parser.add_argument("--clinical-file", default=str(PROCESSED / "clinical.csv"))
    parser.add_argument("--output-dir", default=str(PROCESSED / "cv_run_splits"))
    parser.add_argument("--validation-fraction", type=float, default=0.15,
                        help="Fraction of each outer-training fold reserved for early stopping")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")

    with open(args.folds_file, encoding="utf-8") as f:
        folds = json.load(f)
    clinical = pd.read_csv(args.clinical_file).drop_duplicates("patient_id").set_index("patient_id")
    # The clinical registry contains unlabeled patients outside the audited cohort.
    labels = clinical["label"].dropna().astype(int)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold_index, (fold_name, fold) in enumerate(sorted(folds.items())):
        outer_train = list(fold["train"])
        outer_test = list(fold["val"])
        if set(outer_train) & set(outer_test):
            raise ValueError(f"{fold_name} has patient overlap between train and outer test")
        missing = (set(outer_train) | set(outer_test)) - set(labels.index)
        if missing:
            raise ValueError(f"{fold_name} has patients missing from clinical.csv: {sorted(missing)}")

        y_outer_train = labels.loc[outer_train].to_numpy()
        train_ids, val_ids = train_test_split(
            outer_train,
            test_size=args.validation_fraction,
            random_state=args.seed + fold_index,
            stratify=y_outer_train,
        )
        split = {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(outer_test),
            "metadata": {
                "outer_fold": fold_name,
                "seed": args.seed + fold_index,
                "validation_fraction": args.validation_fraction,
                "outer_test_patients": len(outer_test),
            },
        }
        out_path = output_dir / f"{fold_name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
        print(f"{fold_name}: train={len(train_ids)} val={len(val_ids)} "
              f"outer_test={len(outer_test)} -> {out_path}")


if __name__ == "__main__":
    main()
