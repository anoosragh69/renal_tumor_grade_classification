# vViT and image-baseline comparison — 2026-10-25

The seed-42 vViT and all three image baselines were evaluated on the same held-out patients. Comparisons use one row per patient: hard labels are majority votes over slice predictions, and AUROC scores are mean slice probabilities. DeLong compares paired patient-level ROC curves; exact McNemar compares paired patient correctness.

| Comparator | vViT accuracy | Baseline accuracy | vViT AUROC | Baseline AUROC | DeLong p | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| ViT | 0.774 | 0.806 | 0.611 | 0.732 | 0.574 | 1.000 |
| ConvNeXt | 0.774 | 0.710 | 0.611 | 0.657 | 0.831 | 0.688 |
| ResNeXt | 0.774 | 0.645 | 0.611 | 0.722 | 0.586 | 0.424 |

There is no statistically significant difference in AUROC or paired correctness in these comparisons. This should not be interpreted as equivalence: there are only 31 patients (22 low grade, 9 high grade), so the tests have limited power. The p-values are exploratory and are not adjusted across the three comparisons. Image-level DeLong tests are omitted because thousands of slices from 31 patients are correlated and would give misleading precision.

Generated detailed results (including DeLong z-scores and McNemar discordant-pair counts) are in the git-ignored `results/metrics/vvit_baseline_comparisons_seed42.json` and `.csv` files.

Reproduce with `python -u src/training/compare_vvit_baselines.py` after the prediction artifacts are available.
