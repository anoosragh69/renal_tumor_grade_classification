# Radiomics-only sanity gate (M2)

Logistic regression and random forest on the train-only top-16 radiomic features,
5-fold patient-level stratified CV over the M1 cohort. Top-16 selection (ANOVA
F-value) is fit on each fold's training patients only.

## Patient-level results (primary, mean slice probability per patient)

| Model | Pooled AUROC | Pooled accuracy | n patients |
|---|---|---|---|
| Logistic regression | 0.626 | 0.608 | 171 |
| Random forest | 0.636 | 0.702 | 171 |
| Majority-class baseline | 0.500 | 0.696 | 171 (119 low / 52 high) |

- LR AUROC across folds: mean 0.629 +/- 0.068 (min 0.517, max 0.696)
- RF AUROC across folds: mean 0.641 +/- 0.062 (min 0.567, max 0.746)

## Slice-level results (secondary, inflated by within-patient correlation)

| Model | Mean slice AUROC | Mean slice accuracy |
|---|---|---|
| Logistic regression | 0.575 +/- 0.035 | 0.579 |
| Random forest | 0.555 +/- 0.023 | 0.607 |

## Patient-aggregated variant (features averaged per patient before training)

Tests whether within-patient correlation between slices is what makes the
per-slice gate look near-chance: one row per patient, top-16 selected on the
train patients only.

| Model | Pooled AUROC | Pooled accuracy | n patients |
|---|---|---|---|
| Logistic regression | 0.599 | 0.632 | 171 |
| Random forest | 0.597 | 0.614 | 171 |

## Gate decision

Pooled patient-level AUROC: LR 0.626, RF 0.636. The plan sets the gate at patient-level AUROC near chance (~0.5-0.6), meaning
labels, extraction or leakage must be audited before any neural model is trained.
Accuracy must be read against the majority-class baseline, which is the value a
constant 'low grade' predictor would already achieve.
The paper reports the radiomic sector as the dominant signal (about 0.30
image-level and 0.154 patient-level permutation importance), so some signal was
expected. See `tasks/todo.md` -> Discrepancies.

## Per-fold top-16 features

- Fold 0: original_firstorder_Entropy, original_firstorder_Mean, original_firstorder_Median, original_firstorder_Uniformity, original_glcm_Autocorrelation, original_glcm_ClusterTendency, original_glcm_JointAverage, original_glcm_JointEnergy, original_glcm_MaximumProbability, original_glcm_SumAverage, original_glcm_SumSquares, original_gldm_GrayLevelVariance, original_gldm_HighGrayLevelEmphasis, original_gldm_LargeDependenceHighGrayLevelEmphasis, original_gldm_LargeDependenceLowGrayLevelEmphasis, original_gldm_LowGrayLevelEmphasis
- Fold 1: original_firstorder_Median, original_firstorder_Skewness, original_glcm_Autocorrelation, original_glcm_JointAverage, original_glcm_JointEnergy, original_glcm_JointEntropy, original_glcm_MaximumProbability, original_glcm_SumAverage, original_glcm_SumEntropy, original_gldm_HighGrayLevelEmphasis, original_gldm_LargeDependenceHighGrayLevelEmphasis, original_gldm_LowGrayLevelEmphasis, original_glrlm_HighGrayLevelRunEmphasis, original_glrlm_LowGrayLevelRunEmphasis, original_ngtdm_Busyness, original_ngtdm_Contrast
- Fold 2: original_firstorder_Median, original_glcm_Contrast, original_glcm_DifferenceAverage, original_glcm_Id, original_glcm_Idm, original_glcm_Idmn, original_glcm_Idn, original_glcm_InverseVariance, original_glcm_JointEnergy, original_glcm_MaximumProbability, original_gldm_LargeDependenceHighGrayLevelEmphasis, original_gldm_SmallDependenceLowGrayLevelEmphasis, original_glrlm_RunLengthNonUniformity, original_glrlm_ShortRunHighGrayLevelEmphasis, original_ngtdm_Complexity, original_ngtdm_Contrast
- Fold 3: original_firstorder_Mean, original_firstorder_Median, original_firstorder_Skewness, original_glcm_Autocorrelation, original_glcm_JointAverage, original_glcm_JointEnergy, original_glcm_JointEntropy, original_glcm_MaximumProbability, original_glcm_SumAverage, original_glcm_SumEntropy, original_glcm_SumSquares, original_gldm_GrayLevelVariance, original_gldm_HighGrayLevelEmphasis, original_gldm_LargeDependenceHighGrayLevelEmphasis, original_gldm_LowGrayLevelEmphasis, original_ngtdm_Contrast
- Fold 4: original_firstorder_Mean, original_firstorder_Median, original_firstorder_Uniformity, original_glcm_Autocorrelation, original_glcm_JointAverage, original_glcm_JointEnergy, original_glcm_JointEntropy, original_glcm_MaximumProbability, original_glcm_SumAverage, original_glcm_SumEntropy, original_glcm_SumSquares, original_gldm_GrayLevelVariance, original_gldm_HighGrayLevelEmphasis, original_gldm_LargeDependenceHighGrayLevelEmphasis, original_gldm_LowGrayLevelEmphasis, original_ngtdm_Contrast

## 95% CI (bootstrap over 126 patients, 1000 resamples)

| Model | AUROC | 95% CI |
|---|---|---|
| Logistic regression | 0.626 | 0.533-0.705 |
| Random forest | 0.636 | 0.548-0.720 |
