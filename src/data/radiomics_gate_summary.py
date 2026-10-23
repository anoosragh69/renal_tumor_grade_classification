'''Aggregates per‑fold radiomics gate metrics and computes AUROC statistics'''

import json
from pathlib import Path
import numpy as np

def load_fold_metrics(results_dir):
    metrics = []
    for i in range(5):
        path = results_dir / f"radiomics_gate_fold_{i}.json"
        with open(path, "r") as f:
            metrics.append(json.load(f))
    return metrics

def compute_stats(metrics):
    # Collect AUROC values for each model across folds
    lr_aurocs = []
    rf_aurocs = []
    for m in metrics:
        lr_aurocs.append(m["logistic_regression"]["auroc"])
        rf_aurocs.append(m["random_forest"]["auroc"])
    lr_mean = np.mean(lr_aurocs)
    lr_std = np.std(lr_aurocs, ddof=1)
    rf_mean = np.mean(rf_aurocs)
    rf_std = np.std(rf_aurocs, ddof=1)
    # Simple pooled AUROC approximation: average of per‑fold AUROCs
    pooled_lr = lr_mean
    pooled_rf = rf_mean
    return {
        "logistic_regression": {"mean_auroc": lr_mean, "std_auroc": lr_std, "pooled_auroc": pooled_lr},
        "random_forest": {"mean_auroc": rf_mean, "std_auroc": rf_std, "pooled_auroc": pooled_rf},
    }

def main():
    results_dir = Path(__file__).resolve().parents[2] / "results" / "metrics"
    metrics = load_fold_metrics(results_dir)
    stats = compute_stats(metrics)
    # Write summary JSON
    summary_path = results_dir / "radiomics_gate_auroc_summary.json"
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)
    # Also write a concise markdown note for the todo checkpoint
    note_path = Path(__file__).resolve().parents[2] / "docs" / "radiomics_gate_report.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    with open(note_path, "w") as f:
        f.write("# Radiomics sanity‑gate AUROC summary\n\n")
        f.write(f"Logistic Regression AUROC: {stats['logistic_regression']['mean_auroc']:.3f} ± {stats['logistic_regression']['std_auroc']:.3f}\n\n")
        f.write(f"Random Forest AUROC: {stats['random_forest']['mean_auroc']:.3f} ± {stats['random_forest']['std_auroc']:.3f}\n\n")
        # Simple gate decision
        passed = all(v["mean_auroc"] > 0.6 for v in stats.values())
        f.write(f"Gate decision: {'PASS' if passed else 'FAIL'} (AUROC > 0.6 required)\n")
    print(f"Summary written to {summary_path} and markdown report to {note_path}")

if __name__ == "__main__":
    main()
