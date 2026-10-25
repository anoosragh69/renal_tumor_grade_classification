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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    # Logistic Regression (features are z-scored inside the fold; radiomic
    # features have wildly different scales, and lbfgs fails to converge otherwise)
    lr = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
    ])
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

    metrics = {
        "logistic_regression": {"accuracy": lr_acc, "auroc": lr_auc},
        "random_forest": {"accuracy": rf_acc, "auroc": rf_auc},
    }
    probas = {"logistic_regression": lr_proba, "random_forest": rf_proba}
    return metrics, probas


def to_patient_level(patient_ids, y, proba):
    """
    Aggregate slice-level probabilities to one row per patient (plan 5.2:
    mean slice probability for AUROC, majority vote for the class label).

    Returns (labels, mean_probabilities, sorted patient order).
    """
    frame = pd.DataFrame({"patient_id": patient_ids, "y": y, "p": proba})
    agg = frame.groupby("patient_id", sort=True).agg(y=("y", "first"), p=("p", "mean"))
    return agg["y"].to_numpy(), agg["p"].to_numpy()


def patient_feature_frame(df, feature_cols):
    """Mean radiomic features per patient (one row per patient), label = first."""
    agg = df.groupby("patient_id", sort=True)[feature_cols].mean()
    agg["label"] = df.groupby("patient_id", sort=True)["label"].first()
    return agg[feature_cols], agg["label"].to_numpy()


def main():
    df = load_data()
    folds = load_folds()

    results_dir = Path(__file__).resolve().parents[2] / "results" / "metrics"
    os.makedirs(results_dir, exist_ok=True)

    all_metrics = []
    oof = {"logistic_regression": {"y": [], "p": []}, "random_forest": {"y": [], "p": []}}
    oof_agg = {"logistic_regression": {"y": [], "p": []}, "random_forest": {"y": [], "p": []}}
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

        metrics, probas = train_and_evaluate(X_train, y_train, X_val, y_val)
        metrics["selected_features"] = list(selected_features)
        metrics["fold"] = i

        # Patient-level evaluation (primary, plan 5.2)
        patient_ids = val_df["patient_id"].to_numpy()
        patient_metrics = {}
        for model, proba in probas.items():
            y_p, p_p = to_patient_level(patient_ids, y_val, proba)
            patient_metrics[model] = {
                "accuracy": float(accuracy_score(y_p, (p_p >= 0.5).astype(int))),
                "auroc": float(roc_auc_score(y_p, p_p)),
                "n_patients": int(len(y_p)),
            }
            oof[model]["y"].extend(y_p.tolist())
            oof[model]["p"].extend(p_p.tolist())
        metrics["patient"] = patient_metrics

        # Variant: radiomics averaged per patient *before* training, so the model
        # sees one row per patient instead of correlated slices (tests whether
        # per-slice noise explains a near-chance gate)
        Xtr_p, ytr_p = patient_feature_frame(train_df, feature_cols)
        Xva_p, yva_p = patient_feature_frame(val_df, feature_cols)
        sel_p = select_top_k_features(Xtr_p, ytr_p, k=16)
        agg_metrics, agg_probas = train_and_evaluate(
            Xtr_p[sel_p], ytr_p, Xva_p[sel_p], yva_p)
        agg_metrics["selected_features"] = list(sel_p)
        agg_metrics["n_patients"] = int(len(yva_p))
        for model, proba in agg_probas.items():
            oof_agg[model]["y"].extend(yva_p.tolist())
            oof_agg[model]["p"].extend(proba.tolist())
        metrics["patient_aggregated"] = agg_metrics

        all_metrics.append(metrics)

        fold_path = results_dir / f"radiomics_gate_fold_{i}.json"
        with open(fold_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(
            f"Fold {i}: slice AUROC LR {metrics['logistic_regression']['auroc']:.3f} / "
            f"RF {metrics['random_forest']['auroc']:.3f} | "
            f"patient LR {patient_metrics['logistic_regression']['auroc']:.3f} / "
            f"RF {patient_metrics['random_forest']['auroc']:.3f} | "
            f"patient-agg LR {agg_metrics['logistic_regression']['auroc']:.3f} / "
            f"RF {agg_metrics['random_forest']['auroc']:.3f} "
            f"({len(yva_p)} patients)"
        )

    # Pooled out-of-fold over every patient (each patient is in exactly one val fold)
    def _pooled(oof):
        out = {}
        for model in ("logistic_regression", "random_forest"):
            y = np.asarray(oof[model]["y"])
            p = np.asarray(oof[model]["p"])
            out[model] = {
                "auroc": float(roc_auc_score(y, p)),
                "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
                "n_patients": int(len(y)),
            }
        return out

    pooled = _pooled(oof)
    pooled_agg = _pooled(oof_agg)

    summary = {
        "num_folds": len(folds),
        "per_fold": all_metrics,
        "pooled_patient": pooled,
        "pooled_patient_aggregated": pooled_agg,
    }
    y_all = np.asarray(oof["logistic_regression"]["y"])
    summary["majority_class_baseline"] = {
        "accuracy": float(max(np.mean(y_all == 0), np.mean(y_all == 1))),
        "n_patients": int(len(y_all)),
        "low": int(np.sum(y_all == 0)),
        "high": int(np.sum(y_all == 1)),
    }
    summary_path = results_dir / "radiomics_gate_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregated summary written to {summary_path}")
    for model, m in pooled.items():
        print(f"Pooled patient-level {model}: AUROC {m['auroc']:.3f}, acc {m['accuracy']:.3f}, n={m['n_patients']}")
    for model, m in pooled_agg.items():
        print(f"Pooled patient-AGGREGATED {model}: AUROC {m['auroc']:.3f}, acc {m['accuracy']:.3f}, n={m['n_patients']}")
    base = summary["majority_class_baseline"]
    print(f"Majority-class baseline accuracy: {base['accuracy']:.3f} "
          f"({base['low']} low / {base['high']} high of {base['n_patients']} patients)")

    write_report(summary)


def write_report(summary):
    """Human-readable gate report (M2 checkpoint)."""
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    docs_dir.mkdir(exist_ok=True)
    folds = summary["per_fold"]
    lines = [
        "# Radiomics-only sanity gate (M2)",
        "",
        "Logistic regression and random forest on the train-only top-16 radiomic features,",
        "5-fold patient-level stratified CV over the M1 cohort. Top-16 selection (ANOVA",
        "F-value) is fit on each fold's training patients only.",
        "",
        "## Patient-level results (primary, mean slice probability per patient)",
        "",
        "| Model | Pooled AUROC | Pooled accuracy | n patients |",
        "|---|---|---|---|",
    ]
    for model, m in summary["pooled_patient"].items():
        name = "Logistic regression" if model == "logistic_regression" else "Random forest"
        lines.append(f"| {name} | {m['auroc']:.3f} | {m['accuracy']:.3f} | {m['n_patients']} |")
    base = summary.get("majority_class_baseline", {})
    if base:
        lines.append(
            f"| Majority-class baseline | 0.500 | {base['accuracy']:.3f} | {base['n_patients']} "
            f"({base['low']} low / {base['high']} high) |"
        )
    lr = [f["patient"]["logistic_regression"]["auroc"] for f in folds]
    rf = [f["patient"]["random_forest"]["auroc"] for f in folds]
    lines += [
        "",
        f"- LR AUROC across folds: mean {np.mean(lr):.3f} +/- {np.std(lr):.3f} (min {min(lr):.3f}, max {max(lr):.3f})",
        f"- RF AUROC across folds: mean {np.mean(rf):.3f} +/- {np.std(rf):.3f} (min {min(rf):.3f}, max {max(rf):.3f})",
        "",
        "## Slice-level results (secondary, inflated by within-patient correlation)",
        "",
        "| Model | Mean slice AUROC | Mean slice accuracy |",
        "|---|---|---|",
    ]
    for model in ("logistic_regression", "random_forest"):
        aucs = [f[model]["auroc"] for f in folds]
        accs = [f[model]["accuracy"] for f in folds]
        name = "Logistic regression" if model == "logistic_regression" else "Random forest"
        lines.append(f"| {name} | {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} | {np.mean(accs):.3f} |")
    lines += [
        "",
        "## Patient-aggregated variant (features averaged per patient before training)",
        "",
        "Tests whether within-patient correlation between slices is what makes the",
        "per-slice gate look near-chance: one row per patient, top-16 selected on the",
        "train patients only.",
        "",
        "| Model | Pooled AUROC | Pooled accuracy | n patients |",
        "|---|---|---|---|",
    ]
    for model, m in summary.get("pooled_patient_aggregated", {}).items():
        name = "Logistic regression" if model == "logistic_regression" else "Random forest"
        lines.append(f"| {name} | {m['auroc']:.3f} | {m['accuracy']:.3f} | {m['n_patients']} |")
    lines += [
        "",
        "## Gate decision",
        "",
        f"Pooled patient-level AUROC: LR {summary['pooled_patient']['logistic_regression']['auroc']:.3f}, "
        f"RF {summary['pooled_patient']['random_forest']['auroc']:.3f}. "
        "The plan sets the gate at patient-level AUROC near chance (~0.5-0.6), meaning",
        "labels, extraction or leakage must be audited before any neural model is trained.",
        "Accuracy must be read against the majority-class baseline, which is the value a",
        "constant 'low grade' predictor would already achieve.",
        "The paper reports the radiomic sector as the dominant signal (about 0.30",
        "image-level and 0.154 patient-level permutation importance), so some signal was",
        "expected. See `tasks/todo.md` -> Discrepancies.",
        "",
        "## Per-fold top-16 features",
        "",
    ]
    for f in folds:
        lines.append(f"- Fold {f['fold']}: {', '.join(f['selected_features'])}")
    report = "\n".join(lines) + "\n"
    out = docs_dir / "radiomics_gate_report.md"
    out.write_text(report, encoding="utf-8")
    (docs_dir / "radiomics_gate_auroc_report.md").write_text(report, encoding="utf-8")
    print(f"Report written to {out}")

if __name__ == "__main__":
    main()
