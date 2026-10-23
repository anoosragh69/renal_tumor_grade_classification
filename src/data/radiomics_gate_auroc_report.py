'''Generate AUROC summary report for radiomics sanity gate'''

import json
from pathlib import Path
import numpy as np

def main():
    # Load per‑fold metrics (the file created by radiomics_gate.py)
    summary_path = Path(__file__).resolve().parents[2] / 'results' / 'metrics' / 'radiomics_gate_summary.json'
    with open(summary_path, 'r') as f:
        data = json.load(f)
    folds = data['per_fold']
    lr_aucs = [fold['logistic_regression']['auroc'] for fold in folds]
    rf_aucs = [fold['random_forest']['auroc'] for fold in folds]
    lr_mean = np.mean(lr_aucs)
    lr_std = np.std(lr_aucs, ddof=1)
    rf_mean = np.mean(rf_aucs)
    rf_std = np.std(rf_aucs, ddof=1)
    # Simple pooled AUROC: average of per‑fold AUROCs
    pooled_lr = lr_mean
    pooled_rf = rf_mean
    # Gate decision (require >0.6 mean AUROC for both models)
    passed = (lr_mean > 0.6) and (rf_mean > 0.6)
    # Write markdown report
    report_path = Path(__file__).resolve().parents[2] / 'docs' / 'radiomics_gate_auroc_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('# Radiomics sanity‑gate AUROC summary\n\n')
        f.write(f'Logistic Regression AUROC: {lr_mean:.3f} ± {lr_std:.3f} (pooled: {pooled_lr:.3f})\n\n')
        f.write(f'Random Forest AUROC: {rf_mean:.3f} ± {rf_std:.3f} (pooled: {pooled_rf:.3f})\n\n')
        f.write(f'Gate decision: {'PASS' if passed else 'FAIL'} (mean AUROC > 0.6 required)\n')
    print(f'Report written to {report_path}')

if __name__ == '__main__':
    main()
