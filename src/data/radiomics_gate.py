'''Radiomics sanity‑gate script

This script performs a sanity check for radiomics‑only models using a 5‑fold
patient‑level cross‑validation. For each fold it:

1. Loads the radiomics feature matrix and the corresponding labels.
2. Selects the top‑16 features based on ANOVA F‑test **using only the training
   data of that fold** (to avoid leakage).
3. Trains a Logistic Regression (balanced class weight) and a Random Forest
   (balanced class weight) on the selected features.
4. Evaluates accuracy and AUROC on the validation set.
5. Saves per‑fold metrics and the selected feature names to
   `results/metrics/radiomics_gate_fold_{i}.json`.
6. After all folds, aggregates the metrics and writes a summary JSON file.

The script expects the following files (relative to the project root):
- `data/processed/radiomics_all.csv` – contains columns `patient_id`, `slice_idx`,
  `label` and the radiomics feature columns.
- `data/processed/cv_folds.json` – dictionary where each key is ``"fold_i"`` and the
  value is a dict with ``"train"`` and ``"val"`` lists of patient IDs.

Results are written under `results/metrics/` which is created if it does not
exist.
''' 

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load radiomics data CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with patient_id, slice_idx, label and feature columns.
    """
    csv_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "radiomics_all.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Radiomics CSV not found at {csv_path}")
    return pd.read_csv(csv_path)


def load_folds() -> list:
    """Load CV folds definition from ``cv_folds.json``.

    The JSON file is a dict where keys are ``"fold_0"``, ``"fold_1"`` … and each
    value contains ``"train"`` and ``"val"`` lists of patient IDs.
    This function returns a **list** ordered by the fold number for deterministic
    processing.
    """
    folds_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "cv_folds.json"
    with open(folds_path, "r") as f:
        folds_dict = json.load(f)
    # Sort keys to maintain order ``fold_0``, ``fold_1``, …
    sorted_items = sorted(folds_dict.items(), key=lambda kv: kv[0])
    # Return just the dict values (each containing train/val)
    return [value for _, value in sorted_items]


def select_top_k_features(X_train: pd.DataFrame, y_train: np.ndarray, k: int = 16):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)
    mask = selector.get_support()
    selected_features = X_train.columns[mask]
    return selected_features


def train_and_evaluate(X_train, y_train, X_val, y_val):
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, solver="lbfgs")
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_val)
    lr_proba = lr.predict_proba(X_val)[:, 1]
    lr_acc = accuracy_score(y_val, lr_pred)
    lr_auc = roc_auc_score(y_val, lr_proba)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    rf_proba = rf.predict_proba(X_val)[:, 1]
    rf_acc = accuracy_score(y_val, rf_pred)
    rf_auc = roc_auc_score(y_val, rf_proba)

    return {
        "logistic_regression": {"accuracy": lr_acc, "auroc": lr_auc},
        "random_forest": {"accuracy": rf_acc, "auroc": rf_auc},
    }


def main():
    df = load_data()
    folds = load_folds()

    results_dir = Path(__file__).resolve().parents[2] / "results" / "metrics"
    os.makedirs(results_dir, exist_ok=True)

    all_metrics = []
    for i, fold in enumerate(folds):
        train_patients = set(fold["train"])  # type: ignore[index]
        val_patients = set(fold["val"])      # type: ignore[index]
        train_df = df[df["patient_id"].isin(train_patients)]
        val_df = df[df["patient_id"].isin(val_patients)]

        y_train = train_df["label"].values
        y_val = val_df["label"].values

        feature_cols = [c for c in df.columns if c not in {"patient_id", "slice_idx", "label"}]
        X_train_full = train_df[feature_cols]
        X_val_full = val_df[feature_cols]

        selected_features = select_top_k_features(X_train_full, y_train, k=16)
        X_train = X_train_full[selected_features]
        X_val = X_val_full[selected_features]

        metrics = train_and_evaluate(X_train, y_train, X_val, y_val)
        metrics["selected_features"] = list(selected_features)
        metrics["fold"] = i
        all_metrics.append(metrics)

        fold_path = results_dir / f"radiomics_gate_fold_{i}.json"
        with open(fold_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Fold {i} results written to {fold_path}")

    summary = {"num_folds": len(folds), "per_fold": all_metrics}
    summary_path = results_dir / "radiomics_gate_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregated summary written to {summary_path}")

if __name__ == "__main__":
    main()
