"""Patient-level 95% CI (bootstrap over patients) for the M2 radiomics gate.

Repeats the gate's 5-fold CV, aggregates slice probabilities to one value per
patient, then bootstraps patients (not slices) 1000x for the AUROC CI, as the
plan requires for every reported metric.

Usage: python src/data/radiomics_gate_ci.py
Output: results/metrics/radiomics_gate_ci.json and an appendix in
        docs/radiomics_gate_report.md
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
META = {"patient_id", "slice_idx", "label"}
N_BOOT = 1000


def patient_scores(df, folds):
    """Return {model: (y_patients, oof_mean_probability)} over all patients."""
    out = {"logistic_regression": [], "random_forest": []}
    for fold in folds:
        train = df[df["patient_id"].isin(set(fold["train"]))]
        val = df[df["patient_id"].isin(set(fold["val"]))]
        cols = [c for c in df.columns if c not in META]
        sel = SelectKBest(f_classif, k=16).fit(train[cols], train["label"].values)
        features = list(np.array(cols)[sel.get_support()])
        Xt, yt = train[features], train["label"].values
        models = {
            "logistic_regression": Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]),
            "random_forest": RandomForestClassifier(
                n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42),
        }
        agg = val[["patient_id", "label"]].copy()
        for name, model in models.items():
            model.fit(Xt, yt)
            out[name].append(pd.DataFrame({
                "patient_id": val["patient_id"].values,
                "y": val["label"].values,
                "p": model.predict_proba(val[features])[:, 1],
            }).groupby("patient_id", sort=True).agg(y=("y", "first"), p=("p", "mean")))
    return {name: pd.concat(parts) for name, parts in out.items()}


def bootstrap_ci(y, p, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(roc_auc_score(y, p)), float(lo), float(hi)


def main():
    df = pd.read_csv(PROCESSED / "radiomics_all.csv")
    with open(PROCESSED / "cv_folds.json") as f:
        folds = [v for _, v in sorted(json.load(f).items(), key=lambda kv: kv[0])]

    scores = patient_scores(df, folds)
    results = {}
    for name, agg in scores.items():
        auroc, lo, hi = bootstrap_ci(agg["y"], agg["p"])
        results[name] = {"auroc": auroc, "ci95_low": lo, "ci95_high": hi,
                         "n_patients": int(len(agg))}
        print(f"{name}: AUROC {auroc:.3f} (95% CI {lo:.3f}-{hi:.3f}), n={len(agg)}")

    (ROOT / "results" / "metrics" / "radiomics_gate_ci.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    lines = ["", "## 95% CI (bootstrap over 126 patients, 1000 resamples)", "",
             "| Model | AUROC | 95% CI |", "|---|---|---|"]
    for name, r in results.items():
        label = "Logistic regression" if name == "logistic_regression" else "Random forest"
        lines.append(f"| {label} | {r['auroc']:.3f} | {r['ci95_low']:.3f}-{r['ci95_high']:.3f} |")
    report = ROOT / "docs" / "radiomics_gate_report.md"
    report.write_text(report.read_text(encoding="utf-8") + "\n".join(lines) + "\n",
                      encoding="utf-8")
    print(f"appended CI table to {report}")


if __name__ == "__main__":
    main()
