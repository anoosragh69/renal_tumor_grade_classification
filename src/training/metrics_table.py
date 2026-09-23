"""
M3 checkpoint: paper-style metrics table from Step 9a test evaluation
=====================================================================
Reads results/metrics/test_predictions.csv (produced by
run_evaluation.py) and renders the paper Table 2-style performance table:

  one row per sector head + majority-vote fusion row, columns
  Acc / Sens / Spec / PPV / NPV / F1 / kappa / AUC, each with a
  bootstrap 95% CI (1000x), image-level; plus a patient-level block
  (majority-vote labels, mean-probability scores).

Since our test set is 12 patients (2,227 slices) vs the paper's 30, the
table is the *equivalent* of the paper's Table 2 given our split — counts
are stated in the header, per the deviation policy.

Outputs (git-ignored under results/metrics/):
    table2_style.md    markdown table for the report
    table2_style.csv   same numbers in long format

Usage:
    python src/training/metrics_table.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd

from src.models.vvit import TOKEN_ORDER
from src.training.evaluation import binary_metrics, bootstrap_ci, aggregate_patient
from src.training.run_training import METRICS_DIR

COL_KEYS = ["accuracy", "sensitivity", "specificity", "ppv", "npv", "f1",
            "kappa", "auroc"]
COL_NAMES = {"accuracy": "Acc", "sensitivity": "Sens", "specificity": "Spec",
             "ppv": "PPV", "npv": "NPV", "f1": "F1", "kappa": "Kappa",
             "auroc": "AUC"}


def row_dict(y, pred, score, label, n_boot, seed):
    m = binary_metrics(y, pred, score)
    ci = bootstrap_ci(y, pred, score, n_boot=n_boot, seed=seed)
    out = {"group": label}
    for k in COL_KEYS:
        v = m.get(k)
        lo, hi = ci.get(k, (float("nan"), float("nan")))
        out[k] = v
        out[k + "_ci"] = (lo, hi)
    return out


def fmt(row, key):
    v = row[key]
    lo, hi = row[key + "_ci"]
    if v is None or not np.isfinite(v):
        return "n/a"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{v:.3f}"
    return f"{v:.3f} [{lo:.3f}\u2013{hi:.3f}]"


def to_markdown(rows, title, n_note):
    head = "| " + " | ".join(["Classifier"] + [COL_NAMES[k] for k in COL_KEYS]) + " |"
    sep = "|---|" + "---|" * len(COL_KEYS)
    lines = [f"### {title}", "", f"*{n_note}*", "", head, sep]
    for r in rows:
        lines.append("| " + r["group"] + " | " + " | ".join(fmt(r, k) for k in COL_KEYS) + " |")
    return "\n".join(lines)


def main(n_boot=1000, seed=42):
    pred_path = os.path.join(METRICS_DIR, "test_predictions.csv")
    df = pd.read_csv(pred_path)
    y = df["label"].to_numpy().astype(int)
    n_slices = len(df)
    n_patients = df["patient_id"].nunique()

    # ── Image-level rows: per-sector head + majority-vote fusion ─────────────
    img_rows = []
    for tok in TOKEN_ORDER:
        img_rows.append(row_dict(
            y, df[f"pred_{tok}"].to_numpy(), df[f"prob_{tok}"].to_numpy(),
            tok, n_boot, seed))
    img_rows.append(row_dict(
        y, df["vote_pred"].to_numpy(), df["mean_prob"].to_numpy(),
        "Majority vote (fusion)", n_boot, seed))

    # ── Patient-level rows: fusion only (paper aggregates to patient) ─────────
    agg_vote = aggregate_patient(df["vote_pred"], df["mean_prob"],
                                 df["patient_id"], mode="vote")
    agg_mean = aggregate_patient(df["vote_pred"], df["mean_prob"],
                                 df["patient_id"], mode="mean")
    pids = sorted(agg_vote)
    y_pat = np.array([int(df[df["patient_id"] == pid]["label"].iloc[0]) for pid in pids])
    img_rows_pat = [row_dict(
        y_pat,
        np.array([agg_vote[p][0] for p in pids]),
        np.array([agg_mean[p][1] for p in pids]),
        "Majority vote (fusion)", n_boot, seed)]

    note_img = (f"Image-level: n = {n_slices} tumor slices from {n_patients} test patients; "
                f"95% CI = bootstrap 1000x resample")
    note_pat = (f"Patient-level: n = {n_patients} test patients "
                f"({int((y_pat==0).sum())} low-grade + {int((y_pat==1).sum())} high-grade); "
                f"one prediction per patient (slice majority vote), mean prob for AUC")
    table_img = to_markdown(img_rows, "Table 2-equivalent - image-level performance (vViT, test split)", note_img)
    table_pat = to_markdown(img_rows_pat, "Patient-level performance (vViT, test split)", note_pat)
    md = "# M3 Metrics Table (vViT voting baseline, test split)\n\n" + \
         "*Note: our test split is 12 patients vs the paper's 30 - Table 2-equivalent given our data (deviation documented in plan section 6).*\n\n" + \
         table_img + "\n\n---\n\n" + table_pat + "\n"

    md_path = os.path.join(METRICS_DIR, "table2_style.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    # Long-format CSV
    csv_rows = []
    for level, rows in (("image", img_rows), ("patient", img_rows_pat)):
        for r in rows:
            for k in COL_KEYS:
                lo, hi = r[k + "_ci"]
                csv_rows.append({"level": level, "group": r["group"], "metric": k,
                                 "value": r[k], "ci_lo": lo, "ci_hi": hi})
    csv_path = os.path.join(METRICS_DIR, "table2_style.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(md)
    print(f"Saved -> {md_path}")
    print(f"Saved -> {csv_path}")


if __name__ == "__main__":
    main()
