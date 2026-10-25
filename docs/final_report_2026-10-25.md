# Multimodal vViT for renal tumor grade classification

## Abstract

This project recreated a multimodal renal tumor grade classification pipeline using CT tumor crops, per-slice radiomics, and clinical variables. The goal was methodological learning and transparent evaluation rather than matching reported accuracy. We assembled a 171-patient cohort from KiTS19 and trained the vViT on three patient-level paper-style splits and five-fold patient-level cross-validation. Pooled out-of-fold patient AUROC was 0.608 (95% bootstrap CI 0.510–0.699) and accuracy was 0.620. On the seed-42 test set, permutation importance was largest for clinical sectors rather than radiomics. Paired comparisons with image-only baselines did not show statistically significant differences, but the 31-patient test cohort gives limited power. Results should be treated as exploratory.

## Introduction

Renal tumor grade prediction from preoperative CT may combine image appearance with radiomic and clinical information. This project implements a variable Vision Transformer (vViT) inspired by a multimodal study, with separate input sectors and a voting prediction head. The focus is reproducing the method, identifying data and protocol differences, and reporting patient-level performance without using the test set for tuning.

## Methods

### Cohort and labels

The source data were the public KiTS19 GitHub mirror, containing 210 cases. Among those cases, 172 had a WHO/ISUP grade and complete model-facing clinical variables. Grades 1–2 were labeled low grade and grades 3–4 high grade. Applying the declared CT-series image-count mean ± 1 SD rule removed one additional patient, yielding 171 patients (119 low, 52 high). This image-count interpretation was selected because the source paper does not specify how its image counts were computed and alternative tumor-slice interpretations excluded substantially more patients.

The primary seed-42 split contained 124 training patients, 16 validation patients, and 31 test patients. Two additional stratified split seeds were run with 31 test patients each. Their held-out sets overlap by 4–9 patients, so the three results represent split-seed variability, not independent replications. Five-fold patient-stratified CV evaluated all 171 patients once as outer test data; validation subsets were drawn from each outer training fold.

### Inputs and model

Tumor masks defined per-slice crops resized to 128 × 128. PyRadiomics produced 105 per-slice features; ANOVA F-value selection retained 16 features, selected only on training patients. Radiomic z-scaling and continuous clinical z-scaling used training-split statistics. Clinical inputs comprised age, sex, BMI; peripheral vascular disease, diabetes, chronic kidney disease; and smoking and alcohol categories derived from `kits.json`.

The vViT used six tokens: a learned class token and demographic, comorbidity, habit, radiomic, and image tokens. The custom attention block used 8 heads of dimension 64 (inner attention width 512) projected to an embedding width of 128; the encoder depth was 8 and the MLP width was 32. Six output heads were trained with mean binary cross-entropy. Slice labels used majority vote across the six heads, with the class token breaking 3–3 ties. Patient labels were majority votes across slice labels; patient AUROC used mean slice probability.

The configured vViT protocol used Adam (learning rate 1e-3, no weight decay), up to 200 epochs, mixed precision, image-sector augmentation, and early stopping with patience 15 on validation accuracy. Training balanced classes by image subsampling; validation and test sets remained at their observed prevalence.

### Evaluation

Patient-level metrics are primary because slices from the same patient are correlated. Confidence intervals for AUROC use 1,000 bootstrap resamples of patients. Image-level metrics are secondary. The radiomics-only gate used logistic regression and random forest with fold-local top-16 feature selection. vViT sector importance was measured by 100 patient-wise permutations per sector on the seed-42 test set. Pairwise DeLong and exact McNemar tests compared vViT with image-only baselines on their shared patient test set.

## Results

### Radiomics-only gate

On 171 patients, pooled patient AUROC was 0.626 for logistic regression (95% CI 0.533–0.705) and 0.636 for random forest (0.548–0.720). Random forest accuracy was 0.702, compared with a 0.696 majority-class baseline. The gate therefore indicated weak radiomic signal rather than a strong radiomics-only predictor.

### vViT performance

| Split | Best validation accuracy (epoch) | Patient accuracy | Patient AUROC (95% CI) |
|---|---:|---:|---:|
| Paper-style seed 42 | 0.733 (6) | 0.774 | 0.611 (0.359–0.839) |
| Paper-style seed 43 | 0.631 (7) | 0.581 | 0.646 (0.432–0.833) |
| Paper-style seed 44 | 0.814 (15) | 0.548 | 0.540 (0.321–0.741) |

| CV summary | Patient accuracy | Patient AUROC |
|---|---:|---:|
| Mean ± SD across five folds | 0.619 ± 0.095 | 0.618 ± 0.047 |
| Pooled out-of-fold, n=171 | 0.620 | 0.608 (0.510–0.699) |

The different validation-selected epochs and varying held-out performance show substantial run variability. No hyperparameter search was performed to improve accuracy.

### Image-only baselines and paired comparison

The pretrained ViT, ConvNeXt, and ResNeXt baselines had patient AUROCs of 0.732, 0.657, and 0.722 on the same seed-42 test patients. Paired comparison results were:

| Baseline | vViT AUROC | Baseline AUROC | DeLong p | vViT accuracy | Baseline accuracy | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| ViT | 0.611 | 0.732 | 0.574 | 0.774 | 0.806 | 1.000 |
| ConvNeXt | 0.611 | 0.657 | 0.831 | 0.774 | 0.710 | 0.688 |
| ResNeXt | 0.611 | 0.722 | 0.586 | 0.774 | 0.645 | 0.424 |

None of the paired tests was statistically significant. With 31 patients, this does not establish equivalence.

### Permutation importance

On the seed-42 checkpoint, mean accuracy drop (baseline minus permuted) over 100 repeats was:

| Sector | Image accuracy drop | Patient accuracy drop |
|---|---:|---:|
| Demographic | 0.131 | 0.170 |
| Comorbidity | 0.088 | 0.102 |
| Habit | 0.087 | 0.129 |
| Radiomic | 0.014 | 0.037 |
| Image | 0.007 | 0.026 |

Clinical-sector perturbations reduced accuracy more than radiomic or image perturbations in this run. Thus, this project did not reproduce the paper's radiomics-dominant permutation pattern. The 100 perturbations reuse one fitted model and one test set; their distributions describe sensitivity to the chosen permutations and are not independent cohorts. Mann–Whitney p-values are unadjusted and exploratory.

## Discussion

The pipeline and patient-level evaluation were completed, but predictive signals were weak and variable. The radiomics-only gate was near the class-prevalence accuracy baseline, while vViT results varied across split seeds and CV folds. The seed-42 permutation analysis suggests the fitted model relied more on clinical sectors than radiomic and image sectors, but that finding is conditional on one model and one small test split. It should not be generalized to the population or interpreted as a causal feature attribution.

No further accuracy tuning is planned. The learning objective is met by a traceable patient-level pipeline, multiple split evaluations, an explicit comparison to the baselines, and documentation of where the reproduction diverges from the intended method.

## Limitations

- Test sets contain only 31 patients per paper-style seed; patient-level AUROC intervals are wide and paired tests have low power.
- The three paper-style test sets overlap and are not independent. Five-fold CV provides broader coverage but still has fold-level uncertainty.
- The selected CT-series image-count rule leaves 171 patients rather than the paper's 157 and is an explicit interpretation of an underspecified exclusion rule.
- The public GitHub KiTS19 mirror may differ from the paper's TCIA C4KC-KiTS source.
- Radiomics are computed from 2D per-slice crops rather than whole tumors; slice profiles for permutation are aligned by rank when patient slice counts differ.
- Clinical encodings, diabetes definition, loss aggregation, learning rate, voting tie-break, augmentation, early stopping, and mixed precision include choices not fully specified in the paper.
- Baselines use pretrained ImageNet weights and a learning rate of 1e-4; the vViT uses the documented 1e-3 protocol. These are not capacity-matched models.
- Permutation comparisons and their Mann–Whitney p-values are exploratory; repeated perturbations share the same model and test cohort, and p-values are not multiplicity-adjusted.

## Declared deviations

1. Runtime versions: Python 3.14 and PyTorch 2.14.0+cu126 rather than Python 3.8 / PyTorch 1.7.1.
2. Cohort and split sizes: 171 total (119/52), with seed-42 train/validation/test counts 124/16/31 rather than the paper's 157 and 111/15/30 target.
3. The image-count exclusion uses CT-series slices; the paper's exact counting interpretation could not be reproduced.
4. DM is the OR of uncomplicated and end-organ-damage diabetes fields; smoking and alcohol categories use documented 0/1/2 encodings.
5. Radiomics are per-slice 2D crop features, not whole-tumor 3D features.
6. BCE is averaged across batch and heads; learning rate is 1e-3; a 3–3 vote is broken by the class token.
7. Training uses early stopping and AMP rather than a fixed 200-epoch full-precision run.
8. Class balancing is applied to training crops only; validation and test data keep natural prevalence.
9. Patient-level five-fold CV supplements the multiple paper-style splits.
10. The project uses the GitHub KiTS19 mirror rather than TCIA C4KC-KiTS.
11. Three pretrained image architectures were evaluated, exceeding the optional single-baseline plan; their learning rate was set to 1e-4.

## Reproducibility and artifacts

The pinned environment is in `requirements.txt`. Core code is under `src/`; the split, model, and evaluation decisions are in `tasks/plan.md`. Training checkpoints, prediction CSVs, and machine-readable metrics are generated under git-ignored `results/`. Dated run logs and the full metric tables are in `LOG.md` and:

- `docs/radiomics_gate_report.md`
- `docs/vvit_results_2026-10-25.md`
- `docs/baseline_results_2026-10-25.md`
- `docs/vvit_permutation_importance_2026-10-25.md`
- `docs/vvit_baseline_comparison_2026-10-25.md`

The optional `src/training/demo.py` runs a selected patient from a saved checkpoint's held-out test split and prints per-head votes, the final slice vote, and patient probability. It consumes the existing processed crop, radiomics, and clinical artifacts with the checkpoint's train-only preprocessing statistics; those artifacts were derived upstream from CT, tumor masks, and `kits.json`. Raw CT alone is insufficient. The demo is for research illustration only, not diagnosis.
