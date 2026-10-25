# vViT permutation importance — 2026-10-25

Patient-wise sector permutation was run on the seed-42 checkpoint and its untouched test split (31 patients, 6,348 slices). The checkpoint's reconstructed baseline metrics were checked against the saved evaluation output before perturbations. Each of the five input sectors was permuted 100 times while the other sectors stayed fixed.

For each repeat, whole patient profiles were reassigned without replacement. Demographic, comorbidity, and habit values are patient-level vectors and were copied to all target-patient slices. For radiomic and image sectors, donor slices were aligned to target slices by sorted slice rank using evenly spaced nearest ranks. This gives each patient a donor profile while accommodating different numbers of tumor slices.

Importance is the baseline accuracy minus accuracy after sector permutation. Positive values indicate that disrupting the sector reduced accuracy. The reported values are mean ± sample SD across the 100 permutations.

| Sector | Image accuracy drop | Patient accuracy drop |
|---|---:|---:|
| Demographic | 0.131 ± 0.071 | 0.170 ± 0.072 |
| Comorbidity | 0.088 ± 0.032 | 0.102 ± 0.033 |
| Habit | 0.087 ± 0.055 | 0.129 ± 0.059 |
| Radiomic | 0.014 ± 0.009 | 0.037 ± 0.021 |
| Image | 0.007 ± 0.003 | 0.026 ± 0.016 |

For this checkpoint, disrupting the clinical sectors reduced accuracy more than disrupting radiomic or image inputs. This does not reproduce the paper's radiomics-dominant pattern. The result is descriptive of this one fitted model and test split; it is not evidence that clinical data are generally more predictive. The test set is small, and slice-rank alignment is an approximation when donor and target patients have different slice counts.

Pairwise two-sided Mann–Whitney U comparisons of the 100 image-level and patient-level accuracy-drop distributions are included in the JSON artifact. These are exploratory: all repeats reuse the same 31 patients and model, comparisons are unadjusted for multiplicity, and the repeats are not independent test cohorts. Interpret the effect sizes and variability, not the p-values as population-level inference.

The baseline ROC AUCs were 0.572 at image level and 0.611 at patient level. The patient-level ROC has only 31 observations and is correspondingly coarse.

Generated artifacts (git-ignored):

- `results/metrics/vvit_permutation_importance_seed42.csv` — per-repeat drops
- `results/metrics/vvit_permutation_importance_seed42.json` — summary and pairwise tests
- `results/figures/vvit_permutation_importance_seed42.png` — image- and patient-level boxplots
- `results/figures/vvit_roc_seed42.png` — image- and patient-level ROC curves

Reproduction command: `python -u src/training/permutation_importance.py --n-permutations 100 --seed 42 --output-tag seed42`.
