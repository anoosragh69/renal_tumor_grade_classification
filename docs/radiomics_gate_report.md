# Radiomics-only sanity gate (M2)

Logistic regression and random forest on the train-only top-16 radiomic features,
5-fold patient-level stratified CV over the M1 cohort. Top-16 selection (ANOVA
F-value) is fit on each fold's training patients only.

## Patient-level results (primary, mean slice probability per patient)

| Model | Pooled AUROC | Pooled accuracy | n patients |
|---|---|---|---|
| Logistic regression | 0.534 | 0.571 | 126 |
| Random forest | 0.580 | 0.762 | 126 |
| Majority-class baseline | 0.500 | 0.754 | 126 (95 low / 31 high) |

- LR AUROC across folds: mean 0.530 +/- 0.075 (min 0.439, max 0.640)
- RF AUROC across folds: mean 0.600 +/- 0.026 (min 0.561, max 0.632)

## Slice-level results (secondary, inflated by within-patient correlation)

| Model | Mean slice AUROC | Mean slice accuracy |
|---|---|---|
| Logistic regression | 0.531 +/- 0.050 | 0.551 |
| Random forest | 0.540 +/- 0.014 | 0.678 |

## Gate decision

Pooled patient-level AUROC: LR 0.534, RF 0.580. The plan sets the gate at patient-level AUROC near chance (~0.5-0.6), meaning
labels, extraction or leakage must be audited before any neural model is trained.
Accuracy must be read against the majority-class baseline, which is the value a
constant 'low grade' predictor would already achieve.
The paper reports the radiomic sector as the dominant signal (about 0.30
image-level and 0.154 patient-level permutation importance), so some signal was
expected. See `tasks/todo.md` -> Discrepancies.

## Per-fold top-16 features

- Fold 0: original_shape_Maximum2DDiameterColumn, original_shape2D_Perimeter, original_firstorder_Kurtosis, original_firstorder_Maximum, original_firstorder_Mean, original_firstorder_Median, original_firstorder_Minimum, original_glcm_ClusterShade, original_glrlm_HighGrayLevelRunEmphasis, original_glrlm_LowGrayLevelRunEmphasis, original_glrlm_RunLengthNonUniformity, original_glszm_SmallAreaEmphasis, original_glszm_SmallAreaHighGrayLevelEmphasis, original_glszm_SmallAreaLowGrayLevelEmphasis, original_ngtdm_Busyness, original_ngtdm_Contrast
- Fold 1: original_firstorder_Median, original_firstorder_Minimum, original_glcm_Autocorrelation, original_glcm_JointAverage, original_glcm_JointEnergy, original_glcm_JointEntropy, original_glcm_MaximumProbability, original_glcm_SumAverage, original_gldm_GrayLevelNonUniformity, original_gldm_HighGrayLevelEmphasis, original_gldm_LargeDependenceHighGrayLevelEmphasis, original_gldm_LowGrayLevelEmphasis, original_glrlm_HighGrayLevelRunEmphasis, original_glrlm_LowGrayLevelRunEmphasis, original_ngtdm_Busyness, original_ngtdm_Contrast
- Fold 2: original_firstorder_Energy, original_firstorder_Maximum, original_firstorder_TotalEnergy, original_glcm_Contrast, original_glcm_DifferenceAverage, original_glcm_Id, original_glcm_Idm, original_glcm_Idmn, original_glcm_Idn, original_glcm_InverseVariance, original_gldm_DependenceNonUniformityNormalized, original_glrlm_GrayLevelNonUniformity, original_glrlm_HighGrayLevelRunEmphasis, original_glrlm_LowGrayLevelRunEmphasis, original_glrlm_RunLengthNonUniformity, original_glszm_SizeZoneNonUniformity
- Fold 3: original_firstorder_Maximum, original_firstorder_Median, original_glcm_Autocorrelation, original_glcm_ClusterShade, original_glcm_JointAverage, original_glcm_MaximumProbability, original_glcm_SumAverage, original_gldm_GrayLevelNonUniformity, original_gldm_HighGrayLevelEmphasis, original_gldm_LargeDependenceHighGrayLevelEmphasis, original_gldm_LowGrayLevelEmphasis, original_glrlm_HighGrayLevelRunEmphasis, original_glrlm_LowGrayLevelRunEmphasis, original_glrlm_RunLengthNonUniformity, original_ngtdm_Busyness, original_ngtdm_Contrast
- Fold 4: original_shape2D_Perimeter, original_firstorder_Maximum, original_glcm_Autocorrelation, original_glcm_JointAverage, original_glcm_MaximumProbability, original_glcm_SumAverage, original_gldm_HighGrayLevelEmphasis, original_gldm_LargeDependenceHighGrayLevelEmphasis, original_gldm_LargeDependenceLowGrayLevelEmphasis, original_gldm_LowGrayLevelEmphasis, original_glrlm_HighGrayLevelRunEmphasis, original_glrlm_LowGrayLevelRunEmphasis, original_glszm_SizeZoneNonUniformity, original_glszm_SmallAreaEmphasis, original_glszm_SmallAreaLowGrayLevelEmphasis, original_ngtdm_Busyness

## 95% CI (bootstrap over 126 patients, 1000 resamples)

| Model | AUROC | 95% CI |
|---|---|---|
| Logistic regression | 0.534 | 0.407-0.648 |
| Random forest | 0.580 | 0.459-0.695 |
