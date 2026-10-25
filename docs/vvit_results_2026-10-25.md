# vViT results — 2026-10-25

All runs used the configured vViT protocol (lr 1e-3, patience 15 on validation accuracy, AMP, image-sector augmentation, train-only radiomic feature selection/scaling). No hyperparameter tuning was performed. Each row uses the best validation-accuracy checkpoint and evaluates once on the held-out patients.

## Paper-style patient splits

Each test split contains 31 patients. Their test sets overlap pairwise by 4–9 patients, so report them as split-seed variability, not as three independent replications.

| Seed | Best val accuracy (epoch) | Epochs run | Patient accuracy | Patient AUROC (95% CI) |
|---:|---:|---:|---:|---:|
| 42 | 0.733 (6) | 21 | 0.774 | 0.611 (0.359–0.839) |
| 43 | 0.631 (7) | 22 | 0.581 | 0.646 (0.432–0.833) |
| 44 | 0.814 (15) | 30 | 0.548 | 0.540 (0.321–0.741) |

## Five-fold patient-level cross-validation

Each outer test fold was held out from checkpoint selection. Early stopping used an inner, patient-stratified validation subset drawn from the outer training patients.

| Fold | Best val accuracy (epoch) | Epochs run | Test patients | Patient accuracy | Patient AUROC (95% CI) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.675 (5) | 20 | 35 | 0.714 | 0.678 (0.474–0.862) |
| 1 | 0.683 (12) | 27 | 34 | 0.676 | 0.628 (0.405–0.845) |
| 2 | 0.606 (6) | 21 | 34 | 0.471 | 0.575 (0.326–0.802) |
| 3 | 0.708 (17) | 32 | 34 | 0.588 | 0.567 (0.375–0.751) |
| 4 | 0.680 (5) | 20 | 34 | 0.647 | 0.642 (0.392–0.865) |
| Fold mean ± SD | — | — | — | 0.619 ± 0.095 | 0.618 ± 0.047 |
| Pooled out-of-fold | — | — | 171 | 0.620 | 0.608 (0.510–0.699) |

Confidence intervals use 1,000 patient-level bootstrap resamples. The validation accuracy and loss fluctuate considerably, and the patient-level intervals remain broad. These are learning-project results, not a clinical performance claim. No accuracy-optimization phase is planned.

Full metrics, per-slice predictions, run summaries, and scaler metadata are saved under `results/metrics/` and `results/checkpoints/`; these generated artifacts are git-ignored.
