# Task 5: Data Exploration & Analysis — KiTS19 Dataset

## Dataset Overview
- **Metadata**: 210 cases in `kits.json`
- **Imaging**: 300 case directories with `imaging.nii.gz` files (90 extra imaging-only cases beyond metadata)
- **Target variable**: `tumor_isup_grade` (ISUP grading: 1-4, or null for benign)

---

## ISUP Grade Distribution

| Grade | Cases | Percentage |
|-------|-------|------------|
| 1     | 26    | 15.1%      |
| 2     | 93    | 54.1%      |
| 3     | 39    | 22.7%      |
| 4     | 14    | 8.1%       |
| Null (benign) | 38 | —       |

- **Cases with ISUP grades**: 172
- **Cases with null ISUP grade (benign)**: 38
- **No Grade 5** cases in dataset
- **Imbalance ratio** (max/min): 6.64 (Grade 2 vs Grade 4)

---

## Missing Values Analysis

| Feature                | Missing | Percentage |
|------------------------|---------|------------|
| age_at_nephrectomy     | 0       | 0.0%       |
| gender                 | 0       | 0.0%       |
| body_mass_index        | 0       | 0.0%       |
| tumor_isup_grade       | 38      | 18.1%      |
| radiographic_size      | 0       | 0.0%       |
| pathologic_size        | 0       | 0.0%       |
| tumor_histologic_subtype | 0     | 0.0%       |
| malignant              | 0       | 0.0%       |
| smoking_history        | 0       | 0.0%       |
| surgery_type           | 0       | 0.0%       |
| pathology_t_stage      | 0       | 0.0%       |

**Only `tumor_isup_grade` has missing values.** All other key clinical features are complete.

---

## Numerical Feature Statistics

| Feature              | Mean  | Std   | Min  | Max   |
|----------------------|-------|-------|------|-------|
| age_at_nephrectomy   | 58.4  | 14.4  | 1.0  | 90.0  |
| body_mass_index      | 31.2  | 6.5   | 16.2 | 49.6  |
| radiographic_size    | 4.8   | 3.1   | 1.2  | 16.2  |
| pathologic_size      | 4.7   | 3.2   | 0.8  | 22.5  |

---

## Categorical Feature Distributions

### Gender
- Male: 123 (58.6%)
- Female: 87 (41.4%)

### Tumor Histologic Subtype
| Subtype                | Cases | Percentage |
|------------------------|-------|------------|
| clear_cell_rcc         | 143   | 68.1%      |
| papillary              | 21    | 10.0%      |
| chromophobe            | 19    | 9.0%       |
| oncocytoma             | 10    | 4.8%       |
| angiomyolipoma         | 5     | 2.4%       |
| clear_cell_papillary_rcc | 4   | 1.9%       |
| rcc_unclassified       | 2     | 1.0%       |
| mest                   | 2     | 1.0%       |
| spindle_cell_neoplasm  | 1     | 0.5%       |
| wilms                  | 1     | 0.5%       |
| urothelial             | 1     | 0.5%       |
| multilocular_cystic_rcc| 1     | 0.5%       |

### Malignant
- True (malignant): 192 (91.4%)
- False (benign): 18 (8.6%)

### Surgery Type
- Robotic: 122 (58.1%)
- Open: 60 (28.6%)
- Laparoscopic: 28 (13.3%)

---

## Key Implications for Modeling

1. **Class imbalance**: Grade 2 dominates (54.1%), Grade 4 is rare (8.1%). Use weighted loss, oversampling, or class-balanced batches.

2. **Null ISUP grades**: 38 cases (18.1%) have null `tumor_isup_grade` — these are benign tumors. Decision: exclude from training or treat as separate class.

3. **Grade 5 absence**: No Grade 5 cases exist in this dataset. Target is effectively 4-class (1-4).

4. **Extra imaging cases**: 300 imaging directories but only 210 metadata entries. 90 cases lack clinical data — can only be used for image-only tasks.

5. **Data quality**: All key clinical features except `tumor_isup_grade` are complete — minimal imputation needed.

6. **Feature-rich**: Strong mix of demographics, tumor characteristics, comorbidities, and surgical details for multimodal fusion.
